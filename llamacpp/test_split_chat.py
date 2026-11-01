"""CPU-only controls for stage 2's split_generate (no model, no GPU). Exits 1 if any control fails.

Upstream's own RvChat.generate (executor path) and split_generate run on the same scripted fakes: the circuit's
choices (code_token / circuit_token) and Qwen's choices come from one stream the reference machine produced (a real
calc call, its GO, records, BYTE echoes and HALT newline, then the answer and EOS), checked by a real lock Session.
They must give identical results (tokens, counts, lock error), and the split path must:
- RESET the helper first, prefill the prompt at positions 0..P-1, then send one row per text step at P, P+1, ...;
- never send a chip token's row (rows sent after the prefill = text tokens processed);
- let a circuit-owned token override the helper's choice;
- report a wrong chip token as a lock error (not an answer), stop at the text and computer budgets, and raise when
  the circuit gives up a chip position, exactly as upstream does.

usage (riscv dir on PYTHONPATH, the qwenputer venv, CPU): python test_split_chat.py QWENPUTER_MODEL_DIR
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import split_chat as SC  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import chat as CH  # noqa: E402
from chipformer.rv import core, elf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

CALL = "<tool_call>\n<function=calc>\n<parameter=expression>\n{e}\n</parameter>\n</function>\n</tool_call>"
FIRMWARE = CH.Path(CH.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"
V0, HID = 248320, 3968


def machine():
    m = ref.Machine.from_program(elf.program(FIRMWARE))
    BR.park(m)
    return m


def reference_stream(T, prompt, pieces):
    """Model ids a correct Qwenputer generates: Qwen's text pieces and calls, the reference machine's runs."""
    sess = BR.Session(T, machine())
    sess.history(prompt)
    k = len(sess.toks)
    for kind, text in pieces:
        sess.qwen(T.encode(CALL.format(e=text) if kind == "call" else text))
    return [t - S.TEXT_TOKEN if S.is_text(t) else V0 + t for t in sess.toks[k:]] + [248046]


class World:
    """Who chooses each next token of `gen`: the circuit if it is a chip token or follows a chip input, else Qwen."""

    def __init__(self, gen, corrupt_at=None, give_up_at=None):
        self.qwen, self.owner = [gen[0]], []
        for i in range(len(gen) - 1):
            x, y = gen[i], gen[i + 1]
            if i + 1 == corrupt_at:
                y = y + 1 if y + 1 < V0 + S.VOCAB else y - 1
            if x >= V0:
                self.owner.append(None if i == give_up_at else y)      # code_token at a chip step
            else:
                circ = y >= V0
                self.qwen.append(0 if circ else y)
                self.owner.append(y if circ else None)                  # circuit_token at a text step


def fake_chat(T, world, log):
    c = object.__new__(CH.RvChat)
    c.T, c.v0, c.eos, c.c = T, V0, {248046, 248044}, None
    c.tok = type("Tok", (), {"decode": staticmethod(lambda ids: T.decode_bytes(ids).decode(errors="replace"))})()
    c.code_dims = torch.arange(21)
    c.fast, c.executor, c.verify, c.dev = False, True, False, "cpu"
    q, o = list(world.qwen), list(world.owner)

    def logits_for(t):
        lg = torch.full((V0 + S.VOCAB,), -1.0)
        lg[t] = 1.0
        return lg

    c.model = lambda input_ids=None, use_cache=True: type("O", (), {"logits": logits_for(q.pop(0))[None, None].expand(
        1, input_ids.shape[1], -1), "past_key_values": None})()
    c._exec_start = lambda ids: log.append(("start", len(ids)))
    c._exec_step = lambda t: (log.append(("step", t)), torch.zeros(23))[1]
    c._exec_hc = lambda t, keep: torch.full((HID,), float(t))
    c._qwen_step = lambda hc, cache, pos: (log.append(("qwen", pos, int(hc[0]))), logits_for(q.pop(0)))[1]
    c.circuit_token = lambda hc: o.pop(0)
    c.code_token = lambda code: o.pop(0)
    return c, q


class FakeHelper:
    def __init__(self, q, log):
        self.q, self.log = q, log

    def reset(self):
        self.log.append(("reset",))

    def rows(self, pos, rows):
        self.log.append(("rows", list(pos), [int(r[0]) for r in rows]))
        return [self.q.pop(0), 0, 1, 2, 3], [5.0, 4.0, 3.0, 2.0, 1.0]


def run_both(T, prompt, gen, max_new=8192, max_chip=2_000_000, **wk):
    def lock():
        s = BR.Session(T, machine())
        s.history(prompt)
        return s

    core.snapshot = lambda c, m: None                   # the fakes have no circuit to snapshot into
    out = []
    for which in ("upstream", "split"):
        log = []
        c, q = fake_chat(T, World(gen, **wk), log)
        try:
            if which == "upstream":
                r = CH.RvChat.generate(c, prompt, machine(), max_new, max_chip=max_chip, lock=lock())
            else:
                h = FakeHelper(q, log)
                r = SC.split_generate(c, h, prompt, machine(), max_new, max_chip=max_chip, lock=lock(),
                                      prompt_rows=lambda chat, ids: torch.zeros(len(ids), HID))
            res = {k: r[k] for k in ("tokens", "text_tokens", "chip_tokens", "lock_error")}
        except RuntimeError as e:
            res = {"raised": str(e)}
        out.append((res, log))
    return out


def main():
    T = BR.Text(Path(sys.argv[1]) / "tokenizer.json")
    prompt = T.encode("<|im_start|>user\nWhat is it?<|im_end|>\n<|im_start|>assistant\n")
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    one = reference_stream(T, prompt, [("text", "Sure. "), ("call", "84/18"), ("text", "ANSWER: 14/3")])
    two = reference_stream(T, prompt, [("call", "2+2"), ("text", " and "), ("call", "7-9"), ("text", "ANSWER: 4, -2")])
    P = len(prompt)
    for label, gen, kw in (("one call", one, {}), ("two calls", two, {}), ("text budget", two, {"max_new": 40}),
                           ("computer budget", two, {"max_chip": 300}),
                           ("a wrong chip token", one, {"corrupt_at": next(i for i, t in enumerate(one) if t >= V0) + 5}),
                           ("the circuit gives up", one, {"give_up_at": next(i for i, t in enumerate(one) if t >= V0) + 3})):
        mn, mc = kw.pop("max_new", 8192), kw.pop("max_chip", 2_000_000)
        (ru, lu), (rs, ls) = run_both(T, prompt, gen, mn, mc, **kw)
        check(f"same as upstream generate: {label} ({ {k: (len(v) if k == 'tokens' else v) for k, v in rs.items()} })",
              ru == rs)
        if "raised" in rs:
            continue
        rows = [x for x in ls if x[0] == "rows"]
        ok = ls[0] == ("reset",) and rows[0][1] == list(range(P))
        steps = rows[1:]
        ok = ok and [x[1] for x in steps] == [[P + k] for k in range(len(steps))]
        ok = ok and len(steps) == rs["text_tokens"]
        text_inputs = [t for t in rs["tokens"][:len(rs["tokens"])] if t < V0]
        ok = ok and [x[2][0] for x in steps] == text_inputs[:len(steps)]      # the hc of each text input, in order
        ok = ok and [x[1] for x in lu if x[0] == "qwen"] == [P + k for k in range(len(steps))]
        check(f"  split path: reset, prefill 0..{P - 1}, one text row per text step at P, P+1, ..., no chip rows", ok)
        if label == "one call":
            r_tokens = rs["tokens"]
            check("  the expected stream came out (the circuit's choices won over the helper's dummy 0s)",
                  r_tokens == one and rs["lock_error"] is None)
        if label == "a wrong chip token":
            check(f"  a wrong chip token is a lock error, not an answer ({str(rs['lock_error'])[:60]})",
                  rs["lock_error"] is not None)
    print("SPLIT CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
