"""CPU-only controls for stage 1 (no model, no GPU). Exits 1 if any control fails.

1. generate() on a scripted backend gives exactly run_plain.generate's result on the same script (generated ids, text
   count, call records except timing, stop reason, trace tokens): one call, two calls, a text-budget stop, the exact
   computer-budget edge, an over-long call.
2. The backend sees the prompt at positions 0..P-1, then every generated token (forced response tokens included, in
   the emulator's order) at consecutive positions.
3. helper_client fails closed: a child that exits at once, an error reply, a short reply, non-finite values, an id
   outside the vocabulary, values out of order, no reply within the timeout, a helper that stops reading in the
   middle of a large row request (the send is bounded too), a nonzero exit after END; close() ends a child that
   ignores END.
3b. The production LlamaBackend, on a recording helper: RESET first, the prompt at 0..P-1, then every generated token
   (forced ones included) at P, P+1, ...; a backend with one position off is caught.
4. run_plain_llama.main with a fake helper program: a normal run writes complete rows (exit 0); a helper that dies
   mid-run records harness_error on that question and stops the run (exit 1, no later rows).

usage (riscv dir on PYTHONPATH, the qwenputer venv): python test_run_plain_llama.py BASE_MODEL_DIR SCRATCH
"""
import json, os, shutil, subprocess, sys, textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "inside4b" if (HERE / "inside4b").is_dir() else HERE.parent / "inside4b"))  # deployed copy, or the repo's
import run_plain as RP  # noqa: E402
import run_plain_llama as L  # noqa: E402
from helper_client import Helper, HelperError  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402

CALL = "<tool_call>\n<function=calc>\n<parameter=expression>\n{e}\n</parameter>\n</function>\n</tool_call>"
EOS = 248046


def main():
    base, scratch = Path(sys.argv[1]), Path(sys.argv[2])
    T = BR.Text(base / "tokenizer.json")
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    import torch
    RP.DEVICE = "cpu"

    class FakeModel:                                     # test_streams.py's scripted model
        def __init__(self, script):
            self.script, self.i = list(script), 0

        def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
            t = self.script[min(self.i, len(self.script) - 1)]
            self.i += 1
            logits = torch.full((1, input_ids.shape[1], T.n + 1), -1e9)
            logits[0, -1, t] = 0.0
            return type("Out", (), {"logits": logits, "past_key_values": None})()

    class FakeBackend:
        def __init__(self, script):
            self.script, self.i, self.fed = list(script), 0, []

        def _next(self):
            t = self.script[min(self.i, len(self.script) - 1)]
            self.i += 1
            return t

        def prefill(self, ids):
            self.fed = [(p, t) for p, t in enumerate(ids)]
            return self._next()

        def step(self, t):
            self.fed.append((len(self.fed), t))
            return self._next()

    def strip(res):
        gen, n, calls, stop, used = res
        return gen, n, [{k: v for k, v in c.items() if k != "ms"} for c in calls], stop, used

    text = lambda s: T.encode(s)
    c1, c2 = text(CALL.format(e="2+2")), text(CALL.format(e="(2**64 - 1) // 7"))
    _, need, _ = RP.emulator_response(T, c1, 2_000_000)
    long_call = text(CALL.format(e="+".join(["123456789"] * 40)))
    cases = [("one call", c1 + text("ANSWER: 4") + [EOS], 8192, 2_000_000),
             ("two calls", c1 + text("then") + c2 + text("ANSWER: 2635249153387078802") + [EOS], 8192, 2_000_000),
             ("text budget", c1 + text("x" * 50) * 50, 60, 2_000_000),
             ("computer budget at the edge", c1 + text("ANSWER: 4") + [EOS], 8192, need),
             ("one more than the edge", c1 + text("ANSWER: 4") + [EOS], 8192, need + 1),
             ("an over-long call", long_call + text("ANSWER: ?") + [EOS], 8192, 2_000_000)]
    prompt = [1, 2, 3]
    for label, script, mn, mt in cases:
        old = strip(RP.generate(FakeModel(script), prompt, mn, mt, "R", T))
        fb = FakeBackend(script)
        new = strip(L.generate(fb, prompt, mn, mt, "R", T))
        check(f"same result as run_plain.generate: {label} (stop {new[3]}, {len(new[0])} ids, {len(new[2])} calls)",
              old == new)
        pos_ok = [p for p, _ in fb.fed] == list(range(len(fb.fed)))
        fed_gen = [t for _, t in fb.fed[len(prompt):]]
        check(f"  positions consecutive and every generated token fed: {label}",
              pos_ok and fed_gen == new[0][:len(fed_gen)] and len(fed_gen) >= len(new[0]) - 1)
    forced, _, _ = RP.emulator_response(T, c1, 2_000_000)
    fb = FakeBackend(c1 + text("ANSWER: 4") + [EOS])
    gen = L.generate(fb, prompt, 8192, 2_000_000, "R", T)[0]
    k = len(c1)
    check("the emulator's response (HALT newline included) is fed right after </tool_call>",
          gen[k:k + len(forced)] == list(forced) and forced[-1] == T.NL)

    # 3: the pipe client
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    py = sys.executable

    def prog(name, body):
        p = scratch / name
        p.write_text(f"#!{py}\n" + textwrap.dedent(body))
        p.chmod(0o755)
        return str(p)

    dead = prog("dead.py", "import sys; sys.exit(3)\n")
    errp = prog("err.py", """
        import struct, sys
        sys.stdin.buffer.read(8)
        m = b"model load failed"
        sys.stdout.buffer.write(struct.pack("<iI", 1, len(m)) + m); sys.stdout.buffer.flush()
        """)
    short = prog("short.py", """
        import struct, sys
        sys.stdin.buffer.read(8); sys.stdout.buffer.write(struct.pack("<i", 0)); sys.stdout.buffer.flush()
        sys.stdin.buffer.read(16); sys.stdout.buffer.write(struct.pack("<i", 0) + b"\\0" * 7); sys.stdout.buffer.flush()
        """)
    deaf = prog("deaf.py", "import time\nwhile True: time.sleep(1)\n")
    def replier(name, payload):
        """A fake helper that answers RESET, then replies `payload` (a python bytes expression) to TOKENS."""
        return prog(name, f"""
            import math, struct, sys
            r, w = sys.stdin.buffer, sys.stdout.buffer
            while True:
                hdr = r.read(8)
                if len(hdr) < 8: sys.exit(0)
                t, n = struct.unpack("<II", hdr)
                if t == 3: sys.exit(3)
                if t == 1: w.write(struct.pack("<i", 0)); w.flush(); continue
                r.read(8 * n); w.write({payload}); w.flush()
            """)
    nan = replier("nan.py", 'struct.pack("<i5i5f", 0, 1, 2, 3, 4, 5, *([math.nan] * 5))')
    neg = replier("neg.py", 'struct.pack("<i5i5f", 0, -1, 2, 3, 4, 5, 5, 4, 3, 2, 1)')
    big = replier("big.py", 'struct.pack("<i5i5f", 0, 999999, 2, 3, 4, 5, 5, 4, 3, 2, 1)')
    order = replier("order.py", 'struct.pack("<i5i5f", 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5)')
    good = replier("good.py", 'struct.pack("<i5i5f", 0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1)')
    silent = prog("silent.py", """
        import struct, sys, time
        r, w = sys.stdin.buffer, sys.stdout.buffer
        r.read(8); w.write(struct.pack("<i", 0)); w.flush()
        while True: time.sleep(1)
        """)
    for label, exe, want, kw in (("non-finite values", nan, "bad reply values", {}),
                                 ("an id below 0", neg, "bad reply ids", {}),
                                 ("an id outside the vocabulary", big, "bad reply ids", {"n_vocab": 1000}),
                                 ("values out of order", order, "bad reply values", {}),
                                 ("no reply within the timeout", silent, "not done within", {"timeout": 2.0})):
        h = Helper(exe, "x.gguf", **kw)
        try:
            h.reset()
            h.tokens([0], [5])
            got = "no error"
        except HelperError as e:
            got = str(e)
        h.close()
        check(f"helper client: {label} raises HelperError ({got[:50]})", want in got)
    import time as _t
    import numpy as _np
    stall = prog("stall.py", """
        import struct, sys, time
        sys.stdin.buffer.read(8); sys.stdout.buffer.write(struct.pack("<i", 0)); sys.stdout.buffer.flush()
        while True: time.sleep(1)
        """)
    h = Helper(stall, "x.gguf", timeout=1.0)
    h.reset()
    t0 = _t.monotonic()
    try:
        h.rows(list(range(64)), _np.zeros((64, 3968), "f4"))   # ~1 MB: far more than a pipe holds
        got = "no error"
    except HelperError as e:
        got = str(e)
    took = _t.monotonic() - t0
    h.close(wait=1.0)
    check(f"helper client: a helper that stops reading mid-request raises in {took:.1f} s ({got[:45]}), "
          f"child ended {h.p.poll() is not None}", "sent" in got and took < 3 and h.p.poll() is not None)
    try:
        with Helper(good, "x.gguf") as h:
            h.reset()
            h.tokens([0], [5])
        got = "no error"
    except HelperError as e:
        got = str(e)
    check(f"helper client: a nonzero exit after END raises on a normal exit ({got[:40]})", "exited with code 3" in got)

    class RecHelper:
        """Records what the production backend sends; answers from a script."""
        def __init__(self, script):
            self.script, self.i, self.log = list(script), 0, []

        def reset(self):
            self.log.append(("reset",))

        def tokens(self, pos, ids):
            self.log.append(("tokens", list(pos), list(ids)))
            t = self.script[min(self.i, len(self.script) - 1)]
            self.i += 1
            return [t, 0, 1, 2, 3], [5.0, 4.0, 3.0, 2.0, 1.0]

    def backend_ok(be_cls, script):
        rh = RecHelper(script)
        gen = L.generate(be_cls(rh), prompt, 8192, 2_000_000, "R", T)[0]
        log = rh.log
        P = len(prompt)
        ok = log[0] == ("reset",) and log[1] == ("tokens", list(range(P)), prompt)
        steps = log[2:]
        ok = ok and all(len(x[1]) == 1 and x[1][0] == P + k for k, x in enumerate(steps))
        fed = [x[2][0] for x in steps]
        return ok and fed == gen[:len(fed)] and len(fed) == len(gen) - 1

    class OffBy1(L.LlamaBackend):
        def step(self, t):
            self.pos += 1 if self.pos == 3 + 5 else 0      # one position skipped mid-answer
            return super().step(t)

    script = c1 + text("ANSWER: 4") + [EOS]
    check("production backend: RESET, prompt at 0..P-1, every token at P, P+1, ... (forced included)",
          backend_ok(L.LlamaBackend, script))
    check("production backend control: one skipped position is caught", not backend_ok(OffBy1, script))
    for label, exe, call, want in (("a child that exits at once", dead, "reset", "output ended"),
                                   ("an error reply", errp, "reset", "model load failed"),
                                   ("a short reply", short, "tokens", "output ended")):
        h = Helper(exe, "x.gguf")
        try:
            h.reset()
            if call == "tokens":
                h.tokens([0], [5])
            got = "no error"
        except HelperError as e:
            got = str(e)
        h.close()
        check(f"helper client: {label} raises HelperError ({got[:60]})", want in got)
    h = Helper(deaf, "x.gguf")
    h.close(wait=2.0)
    check("helper client: close() ends a child that ignores END", h.p.poll() is not None)

    # 4: the runner with a fake helper (always answers EOS); and one that dies at its 3rd request
    fake = prog("fake.py", """
        import struct, sys
        n_req, die_at = 0, int(sys.argv[3]) if len(sys.argv) > 3 else -1
        r, w = sys.stdin.buffer, sys.stdout.buffer
        while True:
            hdr = r.read(8)
            if len(hdr) < 8: sys.exit(1)
            t, n = struct.unpack("<II", hdr)
            n_req += 1
            if n_req == die_at: sys.exit(9)
            if t == 3: sys.exit(0)
            if t == 1: w.write(struct.pack("<i", 0)); w.flush(); continue
            r.read(8 * n)
            w.write(struct.pack("<i5i5f", 0, 248046, 0, 1, 2, 3, 0.0, -1.0, -2.0, -3.0, -4.0)); w.flush()
        """)
    qs = scratch / "q.jsonl"
    qs.write_text("".join(json.dumps({"id": f"t-{i}", "group": "1", "prompt": f"What is {i}+{i}?"}) + "\n"
                          for i in range(3)))
    gguf = scratch / "fake.gguf"
    gguf.write_bytes(b"GGUF")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([py, str(HERE / "run_plain_llama.py"), str(base), str(gguf), fake, str(qs),
                        str(scratch / "run-ok")], capture_output=True, text=True, env=env)
    rows = [json.loads(l) for l in open(scratch / "run-ok" / "replies.jsonl")] if r.returncode == 0 else []
    check(f"runner, normal: exit {r.returncode}, {len(rows)} rows, all eos",
          r.returncode == 0 and len(rows) == 3 and all(x["run"]["stop"] == "eos" for x in rows))
    dies = prog("dies.py", f"import os, sys\nos.execv({py!r}, [{py!r}, {fake!r}, sys.argv[1], sys.argv[2], '5'])\n")
    r = subprocess.run([py, str(HERE / "run_plain_llama.py"), str(base), str(gguf), dies, str(qs),
                        str(scratch / "run-dies")], capture_output=True, text=True, env=env)
    rows = [json.loads(l) for l in open(scratch / "run-dies" / "replies.jsonl")]
    check(f"runner, helper dies: exit {r.returncode}, {len(rows)} rows, last has harness_error "
          f"({(rows[-1]['harness_error'] or '')[:50]})",
          r.returncode != 0 and rows and rows[-1]["harness_error"] and len(rows) < 3)
    shutil.rmtree(scratch)
    print("STAGE 1 CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
