"""Do the two calc backends honour one contract? Runs the same inputs through the computer and the host-Python backend
exactly as the MCP server calls them, and compares (outcome, text): outcome is ok / refused / failed. Prints every
difference; exits 1 if any difference is not explained by a DECLARED rule.

usage (riscv dir + this dir on PYTHONPATH, V100 visible): python contract_test.py MODEL_DIR OUT_JSON
       python contract_test.py --selftest   (CPU only: the declared rules accept only the observed triples)
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

TRICKY = ["  42  ", " 1 + 2 ", "1\t+2", "1\n+2", "", " ", "abc", "x", "1 2", "1/0", "7 % 0", "7//0", "0**0", "2**-1",
          "(-3)**2", "-3**2", "2^10", "1.5/2", "1.50*2", "0.1+0.2", "-0.0*5", "gcd(0,5)", "gcd(-4,6)", "gcd(1.5,2)",
          "10//-3", "-7%3", "1e5", "1_000", "+5", "--5", "5-", "()", "((1))", "1,000", "1/3+1/6", "1/2/3", "2**200",
          "10**61", "10**62", "9" * 62 + "+1", "9" * 63, "print(1)", "__import__('os')", "２+２", "١+١", "3 * (4 + 5",
          "gcd(12)", "gcd(1,2,3)", "abs(-3)", "2**3**2", "1.5**2", "4/(-203)*962-40*31-(-37)*(-4)", None, 42,
          "1</tool_call>2", "<think>"]
# Differences accepted on purpose: the computer's real limits (firmware calc.c), declared after the first comparison
# (contract-1.log). Exact observed triples (expression, computer outcome, Python outcome): anything else that differs,
# including the same expression with another outcome, is undeclared. Each reason is a diagnosis from the firmware
# source, not a separately measured proof.
TOO_COMPLEX = ("ok", "error (too complex: split it)")
DECLARED = [
    ("the firmware runs out of registers", "42-848/80/(3/6)-5", TOO_COMPLEX, ("ok", "79/5")),
    ("the firmware runs out of registers", "61*(-450)/694+90-(737/4)-(-32)/(21/6)/1", TOO_COMPLEX,
     ("ok", "-1211201/9716")),
    ("a tab is not a space to the firmware", "1\t+2", ("ok", "error"), ("ok", "3")),
    ("1_000 digit separators are Python-only", "1_000", ("ok", "error"), ("ok", "1000")),
    ("the firmware's ** multiplies through 62-digit products (Python checks only the final value)", "10**61",
     ("ok", "error"), ("ok", "1" + "0" * 61)),
]


def declared(row):
    c, p = tuple(row["computer"]), tuple(row["python"])
    return next((why for why, e, dc, dp in DECLARED if row["expr"] == e and c == dc and p == dp), None)


def selftest():
    """Declared rules must accept exactly the observed triples and nothing else (Codex finding 21's controls)."""
    cases = [({"expr": e, "computer": c, "python": p}, True) for _, e, c, p in DECLARED] + [
        ({"expr": "2+2", "computer": TOO_COMPLEX, "python": ("ok", "4")}, False),
        ({"expr": "2**2", "computer": ("ok", "error"), "python": ("ok", "4")}, False),
        ({"expr": "1\t+2", "computer": ("ok", "error"), "python": ("failed", "raised RuntimeError: x")}, False),
        ({"expr": "10**61", "computer": ("ok", "error"), "python": ("ok", "1")}, False),
        ({"expr": "1_000", "computer": ("failed", "computer check failed"), "python": ("ok", "1000")}, False),
    ]
    wrong = 0
    for row, should in cases:
        ok = bool(declared(row)) == should
        wrong += not ok
        print(f"{'ok ' if ok else 'BAD'} {repr(row['expr'])[:40]:40} {'declared' if declared(row) else 'undeclared'}")
    print("SELFTEST", "PASSED" if not wrong else f"FAILED ({wrong})")
    sys.exit(1 if wrong else 0)


def outcome(backend, expr):
    try:
        result, err, _ = backend.run(expr)
    except Exception as e:  # noqa: BLE001
        return ("failed", f"raised {type(e).__name__}: {e}")
    if err is None:
        return ("ok", result)
    return ("refused", err.split(":", 1)[1].strip()) if err.startswith("refused:") else ("failed", err)


def main():
    from chipformer.rv import calc
    from mcp_server import ComputerBackend, PythonBackend
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
    json.dump({"summary": summary, "declared": [list(x) for x in DECLARED], "rows": rows}, open(out, "w"), indent=1)
    sys.exit(1 if undeclared else 0)


if __name__ == "__main__":
    selftest() if sys.argv[1:2] == ["--selftest"] else main()
