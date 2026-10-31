"""Headless controls for rvview.py (CPU only, no screen, no model). Exits 1 if any control fails.

usage (upstream's experimental/riscv on PYTHONPATH): python test_rvview.py [SAVED_QWENPUTER_REPLIES.jsonl]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rvview as V  # noqa: E402
from chipformer.rv import asm  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402


def main():
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    # function calls and the stack: the built-in 'call' demo (fact(7) and sum(12), recursive)
    s = V.from_program("call")
    sp0 = s.m.x[2]
    low, saw_store = sp0, False
    while s.step():
        low = min(low, s.m.x[2])
        saw_store |= bool(s.last["ram"])
    check(f"call demo halts with s1 = fact(7) = 5040 (got {s.m.x[9]})", s.m.halted and s.m.x[9] == 5040)
    check(f"call demo: a0 = sum(1..12) = 78 (got {s.m.x[10]})", s.m.x[10] == 78)
    check(f"call demo: the stack grew (lowest sp {low:#x} < {sp0:#x}) and came back", low < sp0 and s.m.x[2] == sp0)
    check("call demo: stores to the stack were seen", saw_store)

    # branches, loads/stores and changed-register tracking
    for name in ("loop", "memcpy", "fib", "signext"):
        s = V.from_program(name)
        n = s.run(200_000)
        check(f"program {name}: halts ({n} records, {s.m.steps} instructions)", s.m.halted and s.m.fault is None)

    # disassembly with ABI names
    src = ("_start: addi sp, sp, -16\n sw ra, 12(sp)\n lw s0, 8(sp)\n beq a0, a1, _start\n jal ra, _start\n"
           " slli t0, t1, 3\n mul a2, a3, a4\n lui a5, 0x12345\n")
    prog = asm.assemble(src)
    texts = [V.disasm(w, S.ROM_BASE + 4 * i) for i, w in enumerate(prog.text)]
    want = ["addi sp, sp, -16", "sw ra, 12(sp)", "lw s0, 8(sp)", f"beq a0, a1, {S.ROM_BASE:#x}",
            f"jal ra, {S.ROM_BASE:#x}", "slli t0, t1, 3", "mul a2, a3, a4", "lui a5, 0x12345"]
    for t, w in zip(texts, want):
        check(f"disasm {w!r} (got {t!r})", t == w)

    # the calculator: exact output, the digit file in use, refresh records, custom digit ops disassembled
    s = V.from_calc("123456789*987654321")
    max_digits, ops = 0, set()
    while not s.m.halted:
        word = s.m.rom[(s.m.pc - S.ROM_BASE) >> 2]
        ops.add(V.ref.decode(word).op)
        s.step()
        max_digits = max(max_digits, max(len(V.digits(r)) for r in s.m.bn))
    check(f"calc prints 121932631112635269 (got {bytes(s.m.out)!r})", bytes(s.m.out) == b"121932631112635269\n")
    check(f"calc uses the digit file (largest digit register {max_digits} digits)", max_digits >= 9)
    check(f"calc runs custom digit instructions ({sorted(ops & {'BMAC', 'BMSB', 'BLD', 'BGET', 'BPRD', 'BSCAN'})})",
          {"BMAC", "BLD", "BPRD"} <= ops)
    check(f"calc: records = instructions + refreshes ({s.records} = {s.m.steps} + {s.refreshes})",
          s.records == s.m.steps + s.refreshes)
    # refresh records: the trace inserts one after 63 straight-line instructions (no jump), without running one
    straight = "_start:\n" + "    addi t0, t0, 1\n" * 100 + "    li a7, 93\n    ecall\n"
    s = V.Session(V.ref.Machine.from_program(asm.assemble(straight)), V.REFERENCE, "straight", [])
    kinds = []
    while s.step():
        kinds.append(s.last["kind"])
    check(f"refresh records after 63 straight instructions ({s.refreshes} refresh, t0 = {s.m.x[5]})",
          s.refreshes >= 1 and kinds.index("refresh") == 63 and s.m.x[5] == 100
          and s.records == s.m.steps + s.refreshes)

    # replay of a saved Qwenputer answer: the reference reproduces the saved tool response
    if len(sys.argv) > 1:
        path = sys.argv[1]
        for qid in ("pilot-1-001", "pilot-3-001"):
            r = V.load_row(path, qid)
            _, exprs, saved = V.replay_calls(r)
            for k, (e, want_resp) in enumerate(zip(exprs, saved)):
                s = V.from_replay(path, qid, k)
                s.run(None)
                got = bytes(s.m.out).decode().strip()
                check(f"replay {qid} call {k + 1}: reference output {got[:24]!r} = saved response {want_resp[:24]!r}",
                      got == want_resp.strip())
                check("replay is labelled as a re-run, not the original trace", "not recorded" in s.source)

    # the screen at several sizes: never wider than the terminal, labels always there
    s = V.from_calc("2**64 - 1")
    s.run(500)
    for rows, cols in [(24, 80), (60, 200), (15, 40), (40, 120)]:
        lines = V.text_of(V.render(s, rows, cols, 2), cols)
        check(f"render {cols}x{rows}: {len(lines)} rows, widest {max(len(x) for x in lines)}",
              len(lines) == rows and all(len(x) <= cols for x in lines))
        check(f"render {cols}x{rows}: the source label is visible", any("REFERENCE MACHINE"[:12] in x for x in lines))
    print("VIEWER CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
