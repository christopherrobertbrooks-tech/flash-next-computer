"""FAST-PLAN GPU controls for the token-mode split (V100, inside the lock), on the stage-0 fixture prompts, for one
stock-Qwen GGUF. Exit 1 if any fails.
1. The executor state after the token path's prompt initialization (upstream's ExecRow.start) equals the row path's
   (exec_prompt_rows) on the same prompt and snapshot (head states and lag history, bitwise).
2. A free token-mode run (lock on) ends at EOS or the text budget (a budget stop is a legitimate, failed answer under
   the registration, not a runner fault) with no lock error; and the helper's actual token requests equal an
   **independently reconstructed** history: the prompt ids at 0..P-1, then the text tokens of the stream the RV32
   reference itself produces when fed only the model's own (Qwen-chosen) text tokens (its runs supply the chip
   tokens, BYTE echoes and the HALT newline), at P, P+1, ...; EOS not fed.
3. Reset: the first fixture again at the end gives the same tokens (A, B, ..., A) through the same helper.
usage: python test_split_tokens_gpu.py MODEL_DIR GGUF HELPER FIXTURE_DIR
"""
import functools, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import split_chat as SC  # noqa: E402
from helper_client import Helper  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import chat as CH  # noqa: E402
from chipformer.rv import core, elf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

FIRMWARE = Path(CH.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"
EOS = (248046, 248044)


def parked():
    m = ref.Machine.from_program(elf.program(FIRMWARE))
    BR.park(m)
    return m


class Rec:
    def __init__(self, h):
        self.h, self.sent = h, []

    def reset(self):
        self.sent = []
        return self.h.reset()

    def tokens(self, pos, ids):
        self.sent.append((list(pos), list(ids)))
        return self.h.tokens(pos, ids)

    def rows(self, pos, rows):
        raise AssertionError("rows sent in token mode")


def reconstruct(T, v0, prompt, gen):
    """The text inputs a correct run feeds Qwen, from the reference alone: feed it the generated tokens that aren't
    part of a run it produced (Qwen's own text), and take every text token of its stream after the prompt."""
    ses = BR.Session(T, parked())
    ses.history(prompt)
    k0 = len(ses.toks)
    stream = [t - v0 if t >= v0 else BR.text_token(t) for t in gen]
    i = 0
    while i < len(stream):
        t = stream[i]
        assert S.is_text(t), f"a chip token at {i} the reference didn't produce"
        k = len(ses.toks)
        ses.qwen([t - S.TEXT_TOKEN])
        added = ses.toks[k + 1:]
        assert stream[i + 1:i + 1 + len(added)] == added, f"the run after {i} differs from the reference's"
        i += 1 + len(added)
    full = [x - S.TEXT_TOKEN for x in ses.toks[k0:] if S.is_text(x)]
    return full[:-1] if full and full[-1] in EOS else full


def main():
    model, gguf, helper, fdir = sys.argv[1:5]
    fdir = Path(fdir)
    man = json.loads((fdir / "manifest.json").read_text())
    chat = SC.SplitChat(model)
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    def state():
        return [(st["S"].clone(), st["hist"].clone()) for st in chat.erow.heads]

    def run(ids, h):
        lock = BR.Session(chat.T, parked())
        lock.history(ids)
        lock.qwen = functools.partial(lock.qwen, max_records=man["max_chip"])
        rec = Rec(h)
        r = SC.split_generate(chat, rec, ids, parked(), man["max_new"], max_chip=man["max_chip"], lock=lock,
                              lm_input="tokens")
        return r, rec

    first = None
    with Helper(helper, gguf, 16384, n_vocab=chat.v0) as h:
        for f in man["fixtures"]:
            ids = torch.load(fdir / f"{f['id']}.pt", weights_only=False)["prompt_ids"]
            core.snapshot(chat.c, parked())
            SC.exec_prompt_rows(chat, ids)
            s_rows = state()
            core.snapshot(chat.c, parked())
            chat._exec_start(ids)
            s_tok = state()
            same = all(torch.equal(a, b) and torch.equal(c, d) for (a, c), (b, d) in zip(s_rows, s_tok))
            r, rec = run(ids, h)
            gen = r["tokens"]
            want = reconstruct(chat.T, chat.v0, ids, gen)
            P = len(ids)
            sent_ok = rec.sent[0] == (list(range(P)), ids) and \
                [x for x in rec.sent[1:]] == [([P + k], [t]) for k, t in enumerate(want)]
            stop = ("eos" if gen and gen[-1] in EOS else "text budget" if r["text_tokens"] >= man["max_new"]
                    else "other")
            calls = sum(t == chat.T.TCC for t in gen)
            check(f"{f['id']}: executor state = row path's; stop {stop}, lock {r['lock_error']}, {calls} calls, "
                  f"{len(rec.sent) - 1} text requests {'=' if sent_ok else '!='} the reference-reconstructed history",
                  same and stop in ("eos", "text budget") and r["lock_error"] is None and sent_ok)
            if first is None:
                first = (ids, gen)
        r, _ = run(first[0], h)
        check("reset: the first fixture again after all the others gives the same tokens", r["tokens"] == first[1])
    print("TOKEN-MODE GPU CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
