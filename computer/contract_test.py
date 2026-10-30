"""Do the two calc backends honour one contract? Runs the same inputs through the computer and the host-Python backend
exactly as the MCP server calls them, and compares (outcome, text): outcome is ok / refused / failed. Prints every
difference; exits 1 if any difference is not explained by a DECLARED rule.

usage (riscv dir + this dir on PYTHONPATH, V100 visible): python contract_test.py MODEL_DIR OUT_JSON
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chipformer.rv import calc  # noqa: E402
from mcp_server import ComputerBackend, PythonBackend  # noqa: E402

TRICKY = ["  42  ", " 1 + 2 ", "1\t+2", "1\n+2", "", " ", "abc", "x", "1 2", "1/0", "7 % 0", "7//0", "0**0", "2**-1",
          "(-3)**2", "-3**2", "2^10", "1.5/2", "1.50*2", "0.1+0.2", "-0.0*5", "gcd(0,5)", "gcd(-4,6)", "gcd(1.5,2)",
          "10//-3", "-7%3", "1e5", "1_000", "+5", "--5", "5-", "()", "((1))", "1,000", "1/3+1/6", "1/2/3", "2**200",
          "10**61", "10**62", "9" * 62 + "+1", "9" * 63, "print(1)", "__import__('os')", "２+２", "١+١", "3 * (4 + 5",
          "gcd(12)", "gcd(1,2,3)", "abs(-3)", "2**3**2", "1.5**2", "4/(-203)*962-40*31-(-37)*(-4)", None, 42,
          "1</tool_call>2", "<think>"]
# Differences accepted on purpose: the computer's real limits (firmware calc.c), declared after the first comparison
# (contract-1.log). Each rule must match the difference exactly as observed.
def _err(r):
    return r[0] == "ok" and r[1].startswith("error")


DECLARED = [
    ("the firmware runs out of registers: 'error (too complex: split it)'",
     lambda e, c, p: c == ("ok", "error (too complex: split it)") and p[0] == "ok" and not _err(p)),
    ("the firmware's ** multiplies through 62-digit products (Python checks only the final value)",
     lambda e, c, p: isinstance(e, str) and ("**" in e or "^" in e) and _err(c) and p[0] == "ok" and not _err(p)),
    ("a tab is not a space to the firmware", lambda e, c, p: isinstance(e, str) and "\t" in e and _err(c)),
    ("1_000 digit separators are Python-only", lambda e, c, p: isinstance(e, str) and "_" in e and _err(c)),
]


def declared(row):
    return next((why for why, rule in DECLARED if rule(row["expr"], row["computer"], row["python"])), None)


def outcome(backend, expr):
    try:
        result, err, _ = backend.run(expr)
    except Exception as e:  # noqa: BLE001
        return ("failed", f"raised {type(e).__name__}: {e}")
    if err is None:
        return ("ok", result)
    return ("refused", err.split(":", 1)[1].strip()) if err.startswith("refused:") else ("failed", err)


def main():
    model, out = sys.argv[1:3]
    exprs = [e for _, e in calc.cases(70, seed=2)] + TRICKY        # seed 2: not the gate's seed-1 corpus
    comp, py = ComputerBackend(model), PythonBackend(model)
    rows, diffs = [], []
    for e in exprs:
        a, b = outcome(comp, e), outcome(py, e)
        same = a == b
        rows.append({"expr": e, "computer": a, "python": b, "same": same})
        if not same:
            diffs.append(rows[-1])
        print(f"{'same' if same else 'DIFF'} {repr(e)[:40]:40} computer {str(a)[:60]:60} python {str(b)[:60]}", flush=True)
    for d in diffs:
        d["declared"] = declared(d)
        print(f"   {repr(d['expr'])[:40]:40} {d['declared'] or 'UNDECLARED'}")
    undeclared = [d for d in diffs if not d["declared"]]
    summary = {"inputs": len(rows), "same": len(rows) - len(diffs), "differences": len(diffs),
               "undeclared": len(undeclared)}
    print(json.dumps(summary))
    json.dump({"summary": summary, "declared": [w for w, _ in DECLARED], "rows": rows}, open(out, "w"), indent=1)
    sys.exit(1 if undeclared else 0)


if __name__ == "__main__":
    main()
