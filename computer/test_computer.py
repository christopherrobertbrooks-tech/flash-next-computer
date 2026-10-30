"""Interface proof: the Computer service on a declared corpus, every circuit token checked against the reference
machine, results compared with Python's exact answers (upstream calc.expected).

usage (riscv dir on PYTHONPATH, V100 visible): python test_computer.py MODEL_DIR OUT_JSON
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from computer import Computer  # noqa: E402
from chipformer.rv import calc  # noqa: E402

FIRMWARE = Path(calc.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"


def py(expr):
    try:
        return calc.expected(expr).strip()
    except Exception as e:  # noqa: BLE001
        return f"(python: {type(e).__name__})"


def main():
    comp = Computer(sys.argv[1], FIRMWARE)
    results, t0 = [], time.time()

    def run(group, expr, max_chip=200_000):
        r = comp.call(expr, max_chip=max_chip)
        r["group"], r["python"] = group, py(expr)
        r["agrees_with_python"] = r["result"] is not None and r["result"].strip() == r["python"]
        results.append(r)
        flag = "ok " if r["error"] is None else "ERR"
        print(f"{flag} {group:9} {expr[:50]:50} -> {str(r['result'])[:30]:30} python {r['python'][:30]:30} "
              f"{r['chip_tokens']:>6} chip {r.get('seconds', 0):>6}s {'' if r['error'] is None else r['error']}",
              flush=True)

    # 1. random calls per operation (upstream's own generator), each in a fresh session
    for op, expr in calc.cases(70, seed=1):
        comp.new_session()
        run(f"random-{op}", expr)
    # 2. malformed or edge inputs, fresh sessions
    for expr in ["12+*3", "abc", "", "(1+2", "1/0", "7 / 0", "  42  ", "-5*-5", "1.5*2.25", "2**10", "gcd(0,0)"]:
        comp.new_session()
        run("edge", expr)
    # 3. several calls in one session (the machine stays parked between calls)
    comp.new_session()
    for expr in ["2+2", "123456789*987654321", "84/18", "100//7", "gcd(462,1071)"]:
        run("sequence", expr)
    # 4. long operands
    comp.new_session()
    run("long", "9" * 62 + "*" + "9" * 62)
    comp.new_session()
    run("long", "12345678901234567890123456789012345678901234567890/7")
    # 5. the budget stop
    comp.new_session()
    run("budget", "123456789*987654321", max_chip=50)
    ok = [r for r in results if r["error"] is None]
    summary = {"calls": len(results), "reference_clean": len(ok),
               "agree_with_python": sum(r["agrees_with_python"] for r in results),
               "chip_tokens": sum(r["chip_tokens"] for r in results),
               "seconds": round(time.time() - t0, 1)}
    print(json.dumps(summary))
    json.dump({"summary": summary, "results": results}, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
