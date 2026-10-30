"""Stage 5 scorer (STAGE5-PLAN.md). Runs on Dev-Console (bwrap for the code tests works there; no model runs), with
the arms' run folders copied over; asks the gateway's upstream calc (ssh) to tell a bad expression from a computer
limit. Reads the questions and each arm's replies.jsonl; writes
OUT.json (every question x arm, verdicts, failure layers) and prints the summary and the pre-registered verdict.

usage: python score.py QUESTIONS.jsonl OUT.json A=DIR B=DIR C=DIR [Brep=DIR]
       python score.py --selftest        (no runs needed: the CI, the verdict rule and the classifier on made-up rows)
"""
import json, math, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "computer"))
from stage4_requests import final_answer, value  # noqa: E402  (the exact ANSWER-line parser, stage 4 self-tested)

PRIMARY = ["1", "2", "3", "4"]
MARGIN = 0.05                       # equivalence margin, 5 points
LAYERS = ["transport", "budget", "call format", "backend", "final answer", "reasoning", "harness"]


# ---- statistics ----------------------------------------------------------------------------------------------
def wilson(x, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p = x / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - c) / d, (p + z * z / (2 * n) + c) / d


def newcombe_paired(a, b, c, d):
    """95% CI for p1 - p2 from paired counts (Newcombe 1998, method 10). a: both right; b: first right only;
    c: second right only; d: both wrong. Returns (difference, lower, upper)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den else 0.0
    delta = p1 - p2
    lo = delta - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = delta + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return delta, lo, hi


def verdict(lo, hi):
    if -MARGIN < lo and hi < MARGIN:
        return "equivalent"
    if lo > 0:
        return "computer better"
    if hi < 0:
        return "computer worse"
    return "inconclusive"


# ---- code tests in a bwrap wall --------------------------------------------------------------------------------
RUNNER = """import json, sys, socket, _socket
def _no_network(*a, **k):
    raise OSError('network disabled in the test wall')
socket.socket = socket.create_connection = _socket.socket = _no_network
src = open('/w/code.py').read()
env = {}
exec(compile(src, 'answer', 'exec'), env)
out = []
for t in json.load(open('/w/tests.json')):
    try:
        got = eval(t['call'], env)
        out.append(got == eval(t['expect']))
    except Exception as e:
        out.append(False)
print(json.dumps(out))
"""


def code_block(content):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content or "", re.S)
    return blocks[-1] if blocks else None


def run_tests(src, tests):
    """(all passed, per-test list or an error). Runs on Dev-Console (the gateway's AppArmor refuses bwrap's
    namespaces). bwrap wall: every namespace unshared (no network at the kernel level), read-only /usr, no home or
    other files, 10 s, 1 GB address space; sockets are also replaced inside Python."""
    with tempfile.TemporaryDirectory() as w:
        Path(w, "code.py").write_text(src)
        Path(w, "tests.json").write_text(json.dumps(tests))
        Path(w, "run.py").write_text(RUNNER)
        cmd = ["timeout", "10", "prlimit", "--as=1073741824", "--",
               "bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
               "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--ro-bind", w, "/w", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--chdir", "/tmp", "/usr/bin/python3", "/w/run.py"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            res = json.loads(r.stdout.strip().splitlines()[-1])
            return all(res) and len(res) == len(tests), res
        except (ValueError, IndexError):
            return False, f"exit {r.returncode}: {r.stderr[-300:]}"


# ---- one question x arm ----------------------------------------------------------------------------------------
def calc_says_error(expr):
    """Does upstream's Python calculator also call this expression an error (a bad expression, not a computer
    limit)? Asked on the gateway (upstream's package lives there); None if that can't be answered."""
    cmd = ("cd /mnt/nvme/qwenputer/qwenputer/experimental/riscv && PYTHONPATH=. /mnt/nvme/qwenputer/.venv/bin/python "
           "-c 'import json,sys; from chipformer.rv import calc; e=json.load(sys.stdin); "
           "print(calc.expected(e.replace(chr(10), \"\").strip(\" \")).strip())'")
    try:
        r = subprocess.run(["ssh", "ember-gateway", cmd], input=json.dumps(expr), capture_output=True, text=True,
                           timeout=60)
        return r.stdout.strip().startswith("error") if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def score_one(q, row):
    """Returns a dict: correct, layer (None if correct), answer/tool verdicts and the measurements."""
    d = row.get("raw") or {}
    c = (d.get("choices") or [{}])[0] if isinstance(d, dict) else {}
    m = c.get("message") or {}
    content, finish = m.get("content"), c.get("finish_reason")
    events = m.get("strata_mcp") or []
    calls = row.get("tool_calls") or []
    exprs = [str(x.get("expression")) for x in calls]
    results = [str(x.get("result")) for x in calls if x.get("error") is None]
    out = {"id": q["id"], "group": q["group"], "arm": row["arm"], "seconds": row.get("seconds"), "finish": finish,
           "calls": len(calls), "tool_ms": round(sum(x.get("ms") or 0 for x in calls), 1),
           "chip_tokens": sum(x.get("chip_tokens") or 0 for x in calls),
           "computer_seconds": round(sum(x.get("computer_seconds") or 0 for x in calls), 3),
           "completion_tokens": (d.get("usage") or {}).get("completion_tokens") if isinstance(d, dict) else None,
           "mismatch": any(str(x.get("error") or "").startswith("computer check failed") for x in calls)}
    # correctness
    if q["group"] == "5a":
        src = code_block(content)
        ok, detail = run_tests(src, q["tests"]) if src and finish == "stop" else (False, "no code block")
        out["answer"], out["tests"] = ("met" if ok else "failed"), detail
    else:
        got, raw = final_answer(content)
        ok = finish == "stop" and raw is not None and got == value(q["answer"])
        out["answer"], out["answer_raw"] = ("met" if ok else "failed"), raw
    if q["group"] == "4":                    # how far the chain got: the last intermediate the evidence reached
        evidence = results if row["arm"] in ("B", "C") else re.findall(r"\d+", re.sub(r"[,\s]", "", (content or "") +
                                                                               (m.get("reasoning_content") or "")))
        reached = [i + 1 for i, v in enumerate(q["intermediates"]) if v in evidence]
        out["chain_reached"] = max(reached) if reached else 0
        out["chain_steps"] = q["steps"]
        out["chain_evidence"] = "tool log" if row["arm"] in ("B", "C") else "reply text (weak)"
    out["correct"] = bool(ok)
    # failure layer: first match wins
    layer = None
    err_text = json.dumps(d.get("error")) if isinstance(d, dict) and d.get("error") else ""
    if not ok:
        tool_err = [x for x in calls if x.get("error") is not None]
        bad_result = [x for x in calls if x.get("error") is None and str(x.get("result")).startswith("error")]
        loop = any(exprs.count(e) >= 3 for e in set(exprs))
        if row.get("harness_error"):
            layer = "harness"
        elif row.get("http") not in (200, "timeout") and "out of memory" in err_text:
            layer = "backend"                        # the engine ran out of VRAM next to the tool process
        elif row.get("http") not in (200, "timeout"):
            layer = "transport"
        elif row.get("http") == "timeout" or finish == "length" or loop or \
                any(e.get("event") == "limit" for e in events):
            layer = "budget"
        elif any(str(x["error"]).startswith("refused:") for x in tool_err) or \
                any(calc_says_error(str(x.get("expression"))) is not False for x in bad_result):
            layer = "call format"
        elif tool_err or bad_result:
            layer = "backend"
        elif calls:
            layer = "final answer"
        else:
            layer = "reasoning"
    out["layer"] = layer
    out["tool"] = "used" if calls else "none"
    return out


# ---- the whole comparison --------------------------------------------------------------------------------------
def load_rows(path):
    rows = {}
    for l in open(os.path.join(path, "replies.jsonl")):
        r = json.loads(l)
        rows[r["id"]] = r
    return rows


def summarise(qs, scored, arms, runs):
    S = {}
    by = {(r["id"], r["arm"]): r for r in scored}
    for arm in arms:
        rs = [r for r in scored if r["arm"] == arm]
        S[arm] = {"by_group": {}, "layers": {}, "primary_right": 0, "primary_n": 0}
        for g in PRIMARY + ["5a", "5b"]:
            gr = [r for r in rs if r["group"] == g]
            S[arm]["by_group"][g] = f"{sum(r['correct'] for r in gr)}/{len(gr)}"
        prim = [r for r in rs if r["group"] in PRIMARY]
        S[arm]["primary_right"], S[arm]["primary_n"] = sum(r["correct"] for r in prim), len(prim)
        for r in rs:
            if r["layer"]:
                S[arm]["layers"][r["layer"]] = S[arm]["layers"].get(r["layer"], 0) + 1
        secs = [r["seconds"] for r in prim if r["seconds"] is not None]
        S[arm]["median_seconds_primary"] = statistics.median(secs) if secs else None
        S[arm]["group5_tool_calls"] = sum(r["calls"] for r in rs if r["group"] in ("5a", "5b"))
        S[arm]["mismatches"] = sum(r["mismatch"] for r in rs)
        ch = [r for r in rs if r["group"] == "4"]
        S[arm]["chains_full"] = sum(r.get("chain_reached") == r.get("chain_steps") for r in ch)
        cold = runs[arm].get("warmup-1", {}).get("seconds")
        S[arm]["cold_start_seconds"] = cold
        chip = sum(r["chip_tokens"] for r in rs)
        cs = sum(r["computer_seconds"] for r in rs)
        S[arm]["computer_tokens_per_s"] = round(chip / cs, 1) if cs else None
        S[arm]["tool_ms_total"] = round(sum(r["tool_ms"] for r in rs), 1)
    out = {"arms": S}
    if "B" in arms and "C" in arms:
        ids = [q["id"] for q in qs if q["group"] in PRIMARY]
        a = sum(by[i, "C"]["correct"] and by[i, "B"]["correct"] for i in ids)
        b = sum(by[i, "C"]["correct"] and not by[i, "B"]["correct"] for i in ids)
        c = sum(not by[i, "C"]["correct"] and by[i, "B"]["correct"] for i in ids)
        dd = len(ids) - a - b - c
        delta, lo, hi = newcombe_paired(a, b, c, dd)
        mb, mc = S["B"]["median_seconds_primary"], S["C"]["median_seconds_primary"]
        out["primary"] = {"pairs": {"both right": a, "C only": b, "B only": c, "both wrong": dd},
                          "C_minus_B": round(delta, 4), "ci95": [round(lo, 4), round(hi, 4)],
                          "verdict": verdict(lo, hi),
                          "practical": bool(mb and mc and mc <= 2 * mb and mc <= 60),
                          "median_seconds": {"B": mb, "C": mc}}
    if "A" in arms:
        a_right = sum(by[q["id"], "A"]["correct"] for q in qs if q["group"] in ("5a", "5b"))
        out["group5_tripwire"] = {arm: {"right": sum(by[q["id"], arm]["correct"] for q in qs if q["group"] in
                                                     ("5a", "5b")), "A": a_right,
                                        "flag": sum(by[q["id"], arm]["correct"] for q in qs if q["group"] in
                                                    ("5a", "5b")) < a_right - 2}
                                  for arm in arms if arm != "A"}
    return out


def main():
    qpath, outpath = sys.argv[1:3]
    dirs = dict(x.split("=", 1) for x in sys.argv[3:])
    qs = [json.loads(l) for l in open(qpath) if l.strip()]
    runs = {arm: load_rows(p) for arm, p in dirs.items()}
    scored, missing = [], []
    for arm in ("A", "B", "C"):
        if arm not in runs:
            continue
        for q in qs:
            if q["id"] in runs[arm]:
                scored.append(score_one(q, runs[arm][q["id"]]))
            else:
                missing.append((arm, q["id"]))
    summary = summarise(qs, scored, [a for a in ("A", "B", "C") if a in runs], runs)
    summary["missing"] = missing
    if "Brep" in runs:                     # determinism: the repeat must match B exactly on its subset
        same = [i for i, r in runs["Brep"].items() if not i.startswith("warmup") and i in runs["B"] and
                json.dumps((r["raw"].get("choices") or [{}])[0].get("message", {}).get("content")) ==
                json.dumps((runs["B"][i]["raw"].get("choices") or [{}])[0].get("message", {}).get("content"))]
        n = sum(1 for i in runs["Brep"] if not i.startswith("warmup"))
        summary["determinism_B"] = f"{len(same)}/{n} identical replies"
    json.dump({"summary": summary, "rows": scored}, open(outpath, "w"), indent=1)
    print(json.dumps(summary, indent=1))


def selftest():
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}")

    # Newcombe: symmetric, contains the estimate, narrows with n, and ~95% coverage by simulation
    d, lo, hi = newcombe_paired(140, 5, 5, 10)
    check(f"no difference -> CI around 0 ({lo:.3f}, {hi:.3f})", lo < 0 < hi and abs(lo + hi) < 1e-9)
    d, lo, hi = newcombe_paired(100, 30, 2, 28)
    check(f"clear difference -> CI above 0 ({lo:.3f}, {hi:.3f})", lo > 0 and lo < d < hi)
    d, lo, hi = newcombe_paired(158, 1, 1, 0)
    check(f"almost no disagreement -> equivalent ({lo:.3f}, {hi:.3f})", verdict(lo, hi) == "equivalent")
    d, lo, hi = newcombe_paired(120, 10, 10, 20)
    check(f"12.5% disagreement -> inconclusive ({lo:.3f}, {hi:.3f})", verdict(lo, hi) == "inconclusive")
    check("verdict rule: worse", verdict(-0.2, -0.01) == "computer worse")
    import random
    rng, cover, trials = random.Random(7), 0, 2000
    for _ in range(trials):              # true: B right 80%, C right 85%, 10% disagreement structure
        cnt = [0, 0, 0, 0]
        for _ in range(160):
            u = rng.random()
            cnt[0 if u < 0.775 else 1 if u < 0.85 else 2 if u < 0.875 else 3] += 1
        _, lo, hi = newcombe_paired(*cnt)
        cover += lo <= 0.05 <= hi
    check(f"coverage {cover / trials:.3f} close to 0.95", 0.93 <= cover / trials <= 0.97)

    # classifier on made-up rows
    q = {"id": "t", "group": "1", "answer": "121932631112635269"}

    def row(content, finish="stop", calls=(), http=200, events=(), arm="C", err=None):
        return {"arm": arm, "http": http, "seconds": 1.0, "harness_error": err, "tool_calls": list(calls),
                "raw": {"choices": [{"finish_reason": finish, "message": {"content": content,
                                                                         "strata_mcp": list(events)}}]}}
    good = {"expression": "123456789*987654321", "result": "121932631112635269", "error": None}
    cases = [
        ("right", row("ANSWER: 121932631112635269", calls=[good]), True, None),
        ("right, no tool", row("ANSWER: 121932631112635269", arm="A"), True, None),
        ("wrong, no tool", row("ANSWER: 5", arm="A"), False, "reasoning"),
        ("wrong after a good tool result", row("ANSWER: 5", calls=[good]), False, "final answer"),
        ("cut off", row("ANSWER: 121932631112635269", finish="length", calls=[good]), False, "budget"),
        ("round limit", row("ANSWER: 1", calls=[good], events=[{"event": "limit"}]), False, "budget"),
        ("loop", row("ANSWER: 1", calls=[good] * 3), False, "budget"),
        ("refused call", row("ANSWER: 1", calls=[{"expression": "x" * 300, "result": None,
                                                  "error": "refused: the call is 300 tokens"}]), False, "call format"),
        ("computer error", row("ANSWER: 1", calls=[{"expression": "1+1", "result": None,
                                                    "error": "computer check failed: {}"}]), False, "backend"),
        ("http 503", row(None, finish=None, http=503), False, "transport"),
        ("harness", row(None, finish=None, http=None, err="KeyError"), False, "harness"),
    ]
    for label, r, want_ok, want_layer in cases:
        try:
            s = score_one(q, r)
            check(f"classifier: {label} -> {s['layer'] or 'correct'}", s["correct"] == want_ok and s["layer"] == want_layer)
        except Exception as e:  # noqa: BLE001
            check(f"classifier: {label} raised {e!r}", False)
    print("SELFTEST", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    selftest() if sys.argv[1:2] == ["--selftest"] else main()
