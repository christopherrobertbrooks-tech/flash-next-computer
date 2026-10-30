"""Stage 4 integration check: fixed prompts, temperature 0, MCP on, non-streaming. Every prompt has a declared
expectation, and the run exits 1 unless every one is met:
- arithmetic prompts: at least one tool call, no tool errors, every expected value among the tool results, and the
  final answer (normalised: no spaces, commas, markdown or LaTeX \\frac) contains the expected exact value;
- the no-tool prompt: zero tool calls. Its finish is recorded as "haiku completed" (stop, non-empty) or "known: ran
  out while thinking" (length, empty), separately: that outcome is the model's thinking, not the bridge.
Transport failures (no answer, non-200, no finish reason) fail too.

Saves, per request: the request body, the full raw reply, the tool calls it made (the lines the MCP server appended to
CALLS during that request; requests run one at a time) and the verdict.

usage: python stage4_requests.py BASE_URL OUT_JSON CALLS_JSONL
       python stage4_requests.py --selftest     (no server: proves wrong answers, missing tools and errors fail)
"""
import json, re, sys, time, urllib.error, urllib.request
from fractions import Fraction

MUL, POW = 123456789 * 987654321, (2 ** 64 - 1) // 7
PROD = 98765432123 * 12345678987
FRAC = Fraction(PROD, 7)
PROMPTS = [
    ("mul", "What is 123456789 * 987654321? Use the calc tool, then give the exact answer.",
     {"tool": True, "results": [str(MUL)], "answer": str(MUL)}),
    ("pow-floordiv", "Compute (2**64 - 1) // 7 exactly. Use the calc tool.",
     {"tool": True, "results": [str(POW)], "answer": str(POW)}),
    ("two-step", "Multiply 98765432123 by 12345678987, then divide that product by 7. Use the calc tool for each "
                 "step and give the exact final value as a reduced fraction.",
     {"tool": True, "results": [str(PROD), f"{FRAC.numerator}/{FRAC.denominator}"],
      "answer": f"{FRAC.numerator}/{FRAC.denominator}"}),
    ("no-tool", "Write a three-line haiku about rain. Don't use any tools.", {"tool": False}),
]
MODEL = "qwen3.8-flash-next-coder-iq1_m"


def body(text):
    return {"model": MODEL, "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": 3000, "stream": False, "strata_mcp": True}


def ask(base, b):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(b).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
            code = r.status
    except urllib.error.HTTPError as e:
        d, code = {"error": e.read().decode()[:500]}, e.code
    except (urllib.error.URLError, OSError, ValueError) as e:     # no answer, or not JSON
        d, code = {"error": f"{type(e).__name__}: {e}"}, None
    return code, d, round(time.time() - t, 1)


def normalise(text):
    t = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", text or "")
    return re.sub(r"[\s,*\\]", "", t)


def check(expect, code, d, calls):
    """A list of problems (empty = met) and, for the no-tool prompt, the recorded haiku outcome."""
    problems, note = [], None
    c = (d.get("choices") or [{}])[0] if isinstance(d, dict) else {}
    finish, content = c.get("finish_reason"), (c.get("message") or {}).get("content")
    if code != 200 or finish is None:
        return [f"transport: HTTP {code}, finish {finish}"], note
    bad = [x for x in calls if x.get("error") is not None]
    if bad:
        problems.append(f"tool errors: {[x['error'] for x in bad]}")
    if expect["tool"]:
        if not calls:
            problems.append("no tool call")
        got = [x.get("result") for x in calls]
        missing = [r for r in expect["results"] if r not in got]
        if missing:
            problems.append(f"expected tool results missing: {missing} (got {got})")
        if finish != "stop":
            problems.append(f"finish {finish}, expected stop")
        if not re.search(rf"(?<![\d/]){re.escape(expect['answer'])}(?![\d/])", normalise(content)):
            problems.append(f"final answer lacks {expect['answer']}")
    else:
        if calls:
            problems.append(f"{len(calls)} tool call(s) where none were wanted")
        if finish == "stop" and (content or "").strip():
            note = "haiku completed"
        elif finish == "length" and not (content or "").strip():
            note = "known: ran out while thinking"
        else:
            problems.append(f"unexpected no-tool outcome: finish {finish}, content {content!r:.80}")
    return problems, note


def read_lines(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return []


def main():
    base, out, calls_path = sys.argv[1:4]
    rows = []
    for name, text, expect in PROMPTS:
        before = len(read_lines(calls_path))
        b = body(text)
        code, d, s = ask(base, b)
        calls = read_lines(calls_path)[before:]
        problems, note = check(expect, code, d, calls)
        c = (d.get("choices") or [{}])[0]
        m = c.get("message") or {}
        rows.append({"name": name, "request": b, "expect": expect, "http": code, "seconds": s,
                     "finish": c.get("finish_reason"), "usage": d.get("usage"), "content": m.get("content"),
                     "reasoning_chars": len(m.get("reasoning_content") or ""), "tool_calls": calls,
                     "problems": problems, "note": note, "raw": d})
        print(f"{name:13} {'MET ' if not problems else 'FAIL'} HTTP {code} {s:6.1f}s finish {c.get('finish_reason')} "
              f"calls {len(calls)}{' | ' + note if note else ''}{' | ' + '; '.join(problems) if problems else ''}",
              flush=True)
    json.dump({"replies": rows}, open(out, "w"), indent=1)
    failed = [r["name"] for r in rows if r["problems"]]
    print("INTEGRATION CHECK", "FAILED: " + ", ".join(failed) if failed else "PASSED")
    sys.exit(1 if failed else 0)


def selftest():
    """Wrong answers, missing tools, tool errors and bad transport must fail; the saved good shape must pass."""
    def reply(content, finish="stop"):
        return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}

    def call(result, error=None):
        return {"result": result, "error": error}

    mul, two, haiku = PROMPTS[0][2], PROMPTS[2][2], PROMPTS[3][2]
    frac = f"\\(\\frac{{{PROD}}}{{7}}\\)"
    cases = [  # (label, expect, code, reply, calls, should_pass)
        ("good mul", mul, 200, reply(f"**{MUL}** ({MUL:,})"), [call(str(MUL))], True),
        ("good two-step (LaTeX)", two, 200, reply(f"\\[\\boxed{frac}\\]"), [call(str(PROD)), call(f"{PROD}/7")], True),
        ("good haiku", haiku, 200, reply("rain\non\nroofs"), [], True),
        ("known haiku length", haiku, 200, reply(None, "length"), [], True),
        ("wrong answer, no tool", mul, 200, reply("WRONG; no tool used"), [], False),
        ("wrong final answer", mul, 200, reply("121932631112635268"), [call(str(MUL))], False),
        ("longer wrong number", mul, 200, reply(f"{MUL}7"), [call(str(MUL))], False),
        ("right text, no tool", mul, 200, reply(str(MUL)), [], False),
        ("tool error", mul, 200, reply(str(MUL)), [call(None, "refused: x"), call(str(MUL))], False),
        ("wrong tool result", mul, 200, reply(str(MUL)), [call("1")], False),
        ("second step missing", two, 200, reply(frac), [call(str(PROD))], False),
        ("tool used for haiku", haiku, 200, reply("rain"), [call("4")], False),
        ("HTTP 500", mul, 500, {"error": "server failed"}, [], False),
        ("no finish", mul, 200, {"error": "server failed"}, [], False),
        ("answer cut off", mul, 200, reply(str(MUL), "length"), [call(str(MUL))], False),
    ]
    wrong = 0
    for label, expect, code, d, calls, should in cases:
        problems, _ = check(expect, code, d, calls)
        ok = (not problems) == should
        wrong += not ok
        print(f"{'ok ' if ok else 'BAD'} {label:24} {'passes' if not problems else 'fails: ' + '; '.join(problems)}")
    print("SELFTEST", "PASSED" if not wrong else f"FAILED ({wrong})")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    selftest() if sys.argv[1:2] == ["--selftest"] else main()
