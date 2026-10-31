"""CPU-only controls for the inside-4b runners (no model, no GPU). Exits 1 if any control fails.

1. Arm R's response = every text token the reference machine writes after </tool_call> (the HALT-echo newline
   included), for ordinary values, a firmware limit, an error and an over-long call.
2. Arm B's frame (response_tokens) = R's tokens wherever Python and the firmware agree on the value.
3. B's firmware_expression = the firmware's own capture (run on the reference machine) on synthetic rows, including
   Codex's two cases: exactly 80 rows with no closing '<' ("too long"), and a NUL byte ending the call.
4. A budget too small for the run raises in the emulator (the runner turns that into a computer-budget stop).
5. frames.parse controls (an unanswered call can't supply the answer).

usage (riscv dir on PYTHONPATH, the qwenputer venv, CPU): python test_streams.py BASE_MODEL_DIR
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import frames  # noqa: E402
import run_plain as RP  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

CALL = "<tool_call>\n<function=calc>\n<parameter=expression>\n{e}\n</parameter>\n</function>\n</tool_call>"


class SynthText(BR.Text):
    """The real vocabulary plus fake token ids whose text-buffer rows we choose."""

    def __init__(self, base, rows):
        self.__dict__.update(base.__dict__)
        self.fake = {}
        for i, b in enumerate(rows):
            q = self.n + 10 + i
            self.fake[q] = b
        self.fake_ids = list(self.fake)

    def row_bytes(self, qid):
        if qid in self.fake:
            return self.fake[qid][:S.TEXT_ROW].ljust(S.TEXT_ROW, bytes(1))
        return super().row_bytes(qid)


def main():
    T = BR.Text(Path(sys.argv[1]) / "tokenizer.json")
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    # 1 + 2: R's tokens are the machine's full text output after </tool_call>; B's frame matches where values agree
    for e in ["123456789*987654321", "(2**64 - 1) // 7", "84/18", "42-848/80/(3/6)-5", "12+*3", "9" * 62 + "*" + "9" * 62]:
        ids = T.encode(CALL.format(e=e))
        r_text, trace, records = RP.emulator_response(T, ids, 2_000_000)
        m = RP.ref.Machine.from_program(RP.elf.program(RP.FIRMWARE))
        BR.park(m)
        sess = BR.Session(T, m)
        sess.qwen(ids)
        q = sess.qwen_text()
        tcc = len(q) - 1 - q[::-1].index(T.TCC)
        check(f"R = the machine's text after </tool_call>, newline included: {e[:24]}", r_text == q[tcc + 1:]
              and r_text[-1] == T.NL)
        expr, err = RP.firmware_expression(T, ids[1:-1])
        value = err if err else RP.python_value(expr)
        same_value = T.decode_bytes(r_text).decode().split("<tool_response>")[-1].split("</tool_response>")[0].strip() == value
        if same_value:
            check(f"B's frame = R's tokens where the values agree: {e[:24]}", RP.response_tokens(T, value) == r_text)
        else:
            print(f"    (values differ, as declared: firmware {T.decode_bytes(r_text).decode()[18:52]!r}, Python {value!r})")
        print(f"    {trace} trace tokens, {records} records")

    # 3: B's capture = the firmware's, on synthetic rows
    cases = [("exactly 80 rows, no closing '<' -> too long", [b"<f><p>"] + [b"1"] * 79),
             ("a NUL byte ends the call", [b"<f><p>", b"1", b"\0", b"1", b"</p>"]),
             ("a NUL inside a row ends that row", [b"<f><p>", b"2\x003", b"+4", b"</p>"]),
             ("normal: 79 rows then '<'", [b"<f><p>"] + [b"1"] * 78 + [b"</p>"]),
             ("81 rows: the last overwrites row 80", [b"<f><p>"] + [b"1"] * 79 + [b"</p>"])]
    for label, rows in cases:
        ST = SynthText(T, rows)
        expr, err = RP.firmware_expression(ST, ST.fake_ids)
        mine = err if err else RP.python_value(expr)
        fw_text, _, _ = RP.emulator_response(ST, [T.TCO] + ST.fake_ids + [T.TCC], 2_000_000)
        fw = T.decode_bytes(fw_text).decode().split("<tool_response>")[-1].split("</tool_response>")[0].strip()
        # the firmware evaluates what it captured; compare captures through the values both produce
        check(f"capture matches the firmware: {label} (port {mine[:28]!r}, firmware {fw[:28]!r})", mine == fw)

    # 4: a budget too small raises
    try:
        RP.emulator_response(T, T.encode(CALL.format(e="123456789*987654321")), 10)
        check("a too-small record budget raises", False)
    except RuntimeError:
        check("a too-small record budget raises", True)

    # 5: frames
    t = ("<tool_call>\n<function=calc>\n<parameter=expression>\n2+2\n</parameter>\n</function>\n</tool_call>\n"
         "<tool_response>\n4\n</tool_response>\n<think>\n\n</think>\n\nANSWER: 4<|im_end|>")
    check("frames: a complete call, final text after it", frames.parse(t)["final"] == "ANSWER: 4")
    check("frames: an unanswered call can't supply the answer",
          frames.parse(t + "<tool_call>\n<parameter=expression>\nANSWER: 2\n")["final"] == "")
    print("STREAM CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
