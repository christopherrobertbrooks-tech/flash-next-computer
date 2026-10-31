"""Stage 5 scorer (STAGE5-PLAN.md). Runs on Dev-Console (bwrap works there; no model runs) on the arms' run folders
copied over. Reads the questions and each arm's replies.jsonl; writes OUT.json (every question x arm: verdicts,
failure layer, measurements) and prints the summary with the pre-registered verdict.

usage: python score.py QUESTIONS.jsonl OUT.json A=DIR B=DIR C=DIR [Brep=DIR]
       python score.py --selftest     (interval fixtures, coverage, code-test wall and classifier controls)

It refuses to give a verdict on incomplete or inconsistent data: duplicate or unknown question ids, rows from the
wrong arm, a run made on a different question file (sha256), missing primary pairs, or a determinism repeat whose ids
aren't exactly the registered 40.
"""
import hashlib, json, math, os, re, statistics, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "computer"))
from stage4_requests import final_answer, value  # noqa: E402  (the exact ANSWER-line parser, stage 4 self-tested)

PRIMARY = ["1", "2", "3", "4"]
SENSITIVITY = ["1", "3", "4"]       # descriptive only: the primary set without the limit-seeking group 2
MARGIN = 0.05                       # equivalence margin, 5 points


class DataError(Exception):
    """The run data can't support a verdict (incomplete or inconsistent)."""


# ---- statistics ----------------------------------------------------------------------------------------------
def wilson(x, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 1.0
    p = x / n
    c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - c) / d, (p + z * z / (2 * n) + c) / d


def newcombe_paired(a, b, c, d):
    """95% CI for p1 - p2 from paired counts: Newcombe (1998) method 10, with its correction of a positive covariance
    numerator (ad - bc reduced by n/2, floored at 0), as in statsmodels' confint_proportions_paired.
    a: both right; b: first right only; c: second right only; d: both wrong. Returns (difference, lower, upper)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    num = a * d - b * c
    if num > 0:
        num = max(num - n / 2, 0.0)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = num / math.sqrt(den) if den else 0.0
    delta = p1 - p2
    lo = delta - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = delta + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return delta, lo, hi


def verdict(lo, hi):
    if -MARGIN < lo and hi < MARGIN:
        return "practically equivalent (within 5 points)"
    if lo > 0:
        return "computer better"
    if hi < 0:
        return "computer worse"
    return "inconclusive"


# ---- code tests in a bwrap wall --------------------------------------------------------------------------------
# The child gets only the code and the test calls (never the expected values) and returns plain values; the parent
# checks the exit status and the result schema and compares with the expected values itself. Exact types only (a
# subclass of int with a rigged __eq__ is rejected), so the candidate can't fake a pass without computing the values.
CHILD = r"""import json, math, sys, socket, _socket
def _no_network(*a, **k):
    raise OSError('network disabled in the test wall')
socket.socket = socket.create_connection = _socket.socket = _no_network
def plain(x, depth=0):
    if depth > 30:
        raise TypeError('nested too deep')
    t = type(x)
    if x is None or t in (bool, int, str):
        return x
    if t is float:
        return x if math.isfinite(x) else repr(x)
    if t in (list, tuple):
        return [plain(v, depth + 1) for v in x]
    if t is dict and all(type(k) is str for k in x):
        return {k: plain(v, depth + 1) for k, v in x.items()}
    raise TypeError('not a plain value: ' + t.__name__)
calls = json.load(open('/w/calls.json'))
env = {}
exec(compile(open('/w/code.py').read(), 'answer', 'exec'), env)
out = []
for c in calls:
    try:
        out.append({'ok': True, 'value': plain(eval(c, env))})
    except BaseException as e:
        out.append({'ok': False, 'error': type(e).__name__})
sys.stdout.write('\n' + json.dumps(out) + '\n')
"""
MAX_OUT = 1 << 20


class WallError(Exception):
    """The test wall itself didn't work (a harness failure, not a wrong answer)."""


def plain_expected(x):
    if isinstance(x, tuple):
        x = list(x)
    if isinstance(x, list):
        return [plain_expected(v) for v in x]
    if isinstance(x, dict):
        return {k: plain_expected(v) for k, v in x.items()}
    return x


def strict_equal(got, want):
    """Equal values of exactly the same JSON types, recursively (True is not 1, 3.0 is not 3). Tuples in an expected
    value count as lists (JSON has no tuples): the one equivalence allowed."""
    if type(got) is not type(want):
        return False
    if isinstance(want, list):
        return len(got) == len(want) and all(strict_equal(g, w) for g, w in zip(got, want))
    if isinstance(want, dict):
        return got.keys() == want.keys() and all(strict_equal(got[k], want[k]) for k in want)
    return got == want


def _run_wall(src, calls):
    """(exit status, stdout up to MAX_OUT bytes, whether it overflowed). Output is read as it comes and the process
    killed past MAX_OUT, so a candidate can't fill the scorer's memory. stderr is discarded: it's the candidate's."""
    with tempfile.TemporaryDirectory() as w:
        Path(w, "code.py").write_text(src)
        Path(w, "calls.json").write_text(json.dumps(calls))
        Path(w, "child.py").write_text(CHILD)
        cmd = ["timeout", "10", "prlimit", "--as=1073741824", "--",
               "bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
               "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--ro-bind", w, "/w", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--chdir", "/tmp", "/usr/bin/python3", "/w/child.py"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buf, over = b"", False
        while True:
            chunk = p.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            if len(buf) > MAX_OUT:
                over = True
                p.kill()
                break
        p.stdout.close()
        return p.wait(), buf[:MAX_OUT], over


def run_tests(src, tests):
    """(all passed, details). Dev-Console bwrap wall: every namespace unshared (no network at the kernel level),
    read-only /usr, nothing else of the host, 10 s, 1 GB address space per process, output capped while it's read;
    sockets also replaced in Python. Exit status must be 0, the result one list of {ok, value|error} per test, and
    every value strictly equal (same types) to the expected one. A nonzero exit is the candidate's failure unless the
    trusted canary, run right after, also fails: only then is it a wall (harness) failure. Nothing the candidate
    prints can decide which."""
    code, out, over = _run_wall(src, [t["call"] for t in tests])
    if over:
        return False, "output too large"
    if code != 0:
        wall_canary()                                    # raises WallError if the wall itself is broken
        return False, f"exit {code}"
    try:
        res = json.loads(out.decode(errors="replace").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "no result line"
    if not (isinstance(res, list) and len(res) == len(tests) and
            all(isinstance(x, dict) and isinstance(x.get("ok"), bool) for x in res)):
        return False, "malformed result"
    passed = [x["ok"] and "value" in x and
              strict_equal(x["value"], plain_expected(eval(t["expect"])))  # noqa: S307 (our own tests)
              for x, t in zip(res, tests)]
    return all(passed), passed


def wall_canary():
    """A trusted program that must pass: if it doesn't, the wall itself is broken (WallError, a harness failure)."""
    code, out, _ = _run_wall("def f(x):\n    return x * 2\n", ["f(21)"])
    try:
        ok = code == 0 and json.loads(out.decode().strip().splitlines()[-1]) == [{"ok": True, "value": 42}]
    except (ValueError, IndexError):
        ok = False
    if not ok:
        raise WallError(f"the test wall's canary failed (exit {code})")


# ---- one question x arm ----------------------------------------------------------------------------------------
GEN_LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused .*?, (\d+) generated")


def score_one(q, row):
    """One question in one arm. Hard failures (harness, transport, budget, trace mismatch) decide the outcome whatever
    the final text says; otherwise the final answer (or the unit tests) decides. A tool fault the model recovered from
    (a refusal, a computer limit or error, then a right answer) counts as right and is listed under "recovered"."""
    d = row.get("raw") or {}
    c = (d.get("choices") or [{}])[0] if isinstance(d, dict) else {}
    m = c.get("message") or {}
    content, finish = m.get("content"), c.get("finish_reason")
    events = m.get("strata_mcp") or []
    calls = row.get("tool_calls") or []
    exprs = [str(x.get("expression") if x.get("expression") is not None else x.get("call_text")) for x in calls]
    gens = GEN_LINE.findall(row.get("engine_log") or "")
    out = {"id": q["id"], "group": q["group"], "arm": row["arm"], "seconds": row.get("seconds"), "finish": finish,
           "http": row.get("http"), "calls": len(calls), "tool_ms": round(sum(x.get("ms") or 0 for x in calls), 1),
           "chip_tokens": (row.get("inside") or {}).get("chip_tokens")
                          or sum(x.get("chip_tokens") or x.get("trace_tokens") or 0 for x in calls),
           "computer_seconds": round((row.get("inside") or {}).get("chip_seconds")
                                     or sum(x.get("computer_seconds") or 0 for x in calls), 3),
           "rounds": len(gens), "generated_tokens": sum(int(g[2]) for g in gens),
           "tool_rss_peak_mib": max([x.get("rss_peak_mib") or 0 for x in calls], default=None),
           "fault_kinds": sorted({x.get("error_kind") for x in calls if x.get("error_kind")}),
           "computer_error_results": [x["expression"] for x in calls
                                      if x.get("error") is None and str(x.get("result")).startswith("error")],
           "loop": any(exprs.count(e) >= 3 for e in set(exprs)),
           "mismatch": any(x.get("error_kind") == "trace_mismatch" for x in calls)}
    err_text = json.dumps(d.get("error")) if isinstance(d, dict) and d.get("error") else ""
    # 1. hard failures: they decide, whatever the final text
    hard = None
    if row.get("harness_error"):
        hard = "harness"
    elif row.get("http") not in (200, "timeout") and "out of memory" in err_text:
        hard = "backend"                             # the explicit error text names the cause
    elif row.get("http") not in (200, "timeout"):
        hard = "transport"
    elif row.get("http") == "timeout" or finish == "length" or row.get("tool_calls_open") or \
            any(e.get("event") == "limit" for e in events):
        hard = "budget"
    elif out["mismatch"]:
        hard = "backend"
    # 2. the answer
    if q["group"] == "5a":
        src = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content or "", re.S)
        ok, detail = run_tests(src[-1], q["tests"]) if src and finish == "stop" else (False, "no code block")
        out["answer"], out["tests"] = ("met" if ok else "failed"), detail
    else:
        got, raw = final_answer(content)
        ok = finish == "stop" and raw is not None and got == value(q["answer"])
        out["answer"], out["answer_raw"] = ("met" if ok else "failed"), raw
    out["correct"] = bool(ok) and hard is None
    # 3. the layer of a failure (first match wins)
    layer = hard
    if layer is None and not out["correct"]:
        refs = [x.get("python_reference") for x in calls
                if x.get("error") is None and str(x.get("result")).startswith("error")]
        kinds = out["fault_kinds"]
        if "refused" in kinds or "unknown_tool" in kinds or any(r and r.startswith("error") for r in refs):
            layer = "call format"                    # refused input, or an expression Python also calls an error
        elif any(k in kinds for k in ("budget", "execution_error", "reference_error")) or \
                any(r and not r.startswith("error") for r in refs):
            layer = "backend"                        # the tool failed, or the computer said error where Python didn't
        elif any(r is None for r in refs):
            layer = "unknown"                        # an error result with no recorded Python reference
        elif calls:
            layer = "final answer"
        else:
            layer = "reasoning"
    out["layer"] = layer
    out["recovered"] = (out["fault_kinds"] or out["computer_error_results"]) if out["correct"] else []
    # 4. chains (B and C only): how many steps were seen in order in the tool results
    if q["group"] == "4" and row["arm"] in ("B", "C", "R"):
        results = [str(x.get("result")) for x in calls if x.get("error") is None]
        k = 0
        for r in results:
            if k < len(q["intermediates"]) and r == q["intermediates"][k]:
                k += 1
        pool, seen = list(results), 0
        for v in q["intermediates"]:
            if v in pool:
                pool.remove(v)
                seen += 1
        out["chain_steps"], out["chain_in_order"], out["chain_seen_any_order"] = q["steps"], k, seen
    return out


# ---- data integrity ------------------------------------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_arm(path, arm, qids, qhash, expected_ids):
    """The arm's rows by id, after checking: one row per id, the right arm, known ids, the same question file."""
    shaf = os.path.join(path, "questions.sha256")
    try:
        recorded = open(shaf).read().split()[0]
    except (OSError, IndexError):
        raise DataError(f"{path}: no readable questions.sha256 (no proof of which question file was used)")
    if recorded != qhash:
        raise DataError(f"{path}: run on a different question file")
    rows = {}
    for l in open(os.path.join(path, "replies.jsonl")):
        r = json.loads(l)
        if r["id"] in rows:
            raise DataError(f"{path}: duplicate row for {r['id']}")
        if r.get("arm") != arm[0]:
            raise DataError(f"{path}: row {r['id']} is from arm {r.get('arm')}, expected {arm[0]}")
        if r["id"] not in qids and not r["id"].startswith("warmup-"):
            raise DataError(f"{path}: unknown question id {r['id']}")
        rows[r["id"]] = r
    got = {i for i in rows if not i.startswith("warmup-")}
    return rows, sorted(set(expected_ids) - got), sorted(got - set(expected_ids))


def every4(qs):
    prim = [q for q in qs if q["group"] in PRIMARY]
    return [q["id"] for i, q in enumerate(prim) if i % 4 == 0]


# ---- the whole comparison --------------------------------------------------------------------------------------
def paired(by, ids, first, second):
    a = sum(by[i, first]["correct"] and by[i, second]["correct"] for i in ids)
    b = sum(by[i, first]["correct"] and not by[i, second]["correct"] for i in ids)
    c = sum(not by[i, first]["correct"] and by[i, second]["correct"] for i in ids)
    return a, b, c, len(ids) - a - b - c


def summarise(qs, scored, arms, runs, other=None):
    S = {}
    by = {(r["id"], r["arm"]): r for r in scored}
    for arm in arms:
        rs = [r for r in scored if r["arm"] == arm]
        prim = [r for r in rs if r["group"] in PRIMARY]
        secs = [r["seconds"] for r in prim if r["seconds"] is not None]
        chip, cs = sum(r["chip_tokens"] for r in rs), sum(r["computer_seconds"] for r in rs)
        S[arm] = {
            "by_group": {g: f"{sum(r['correct'] for r in rs if r['group'] == g)}/"
                            f"{sum(1 for r in rs if r['group'] == g)}" for g in PRIMARY + ["5a", "5b"]},
            "primary": f"{sum(r['correct'] for r in prim)}/{len(prim)}",
            "layers": {k: sum(r["layer"] == k for r in rs) for k in sorted({r["layer"] for r in rs if r["layer"]})},
            "recovered": sum(bool(r["recovered"]) for r in rs),
            "loops_flagged": sum(r["loop"] for r in rs),
            "median_seconds_primary": statistics.median(secs) if secs else None,
            "cold_start_seconds": (runs[arm].get("warmup-1") or {}).get("seconds"),
            "group5_tool_calls": sum(r["calls"] for r in rs if r["group"] in ("5a", "5b")),
            "mismatches": sum(r["mismatch"] for r in rs),
            "tool_ms_total": round(sum(r["tool_ms"] for r in rs), 1),
            "computer_tokens_per_s": round(chip / cs, 1) if cs else None,
            "tool_rss_peak_mib": max([r["tool_rss_peak_mib"] or 0 for r in rs], default=None),
            "generated_tokens_total": sum(r["generated_tokens"] for r in rs),
        }
        ch = [r for r in rs if "chain_in_order" in r]
        if ch:
            S[arm]["chains_all_steps_seen_in_order"] = \
                f"{sum(r['chain_in_order'] == r['chain_steps'] for r in ch)}/{len(ch)}"
    out = {"arms": S}
    other = other or ("B" if "B" in arms else "R")   # stage 5: C vs B (frozen default); inside-4b: --compare C,R
    if other in arms and "C" in arms:
        ids = [q["id"] for q in qs if q["group"] in PRIMARY]
        a, b, c, dd = paired(by, ids, "C", other)
        delta, lo, hi = newcombe_paired(a, b, c, dd)
        mb, mc = S[other]["median_seconds_primary"], S["C"]["median_seconds_primary"]
        out["primary"] = {"compared": f"C vs {other}", "questions": len(ids),
                          "pairs": {"both right": a, "C only": b, f"{other} only": c, "both wrong": dd},
                          f"C_minus_{other}": round(delta, 4), "ci95": [round(lo, 4), round(hi, 4)],
                          "verdict": verdict(lo, hi),
                          "practical": bool(mb and mc and mc <= 2 * mb and mc <= 60),
                          "median_seconds": {other: mb, "C": mc}}
        sids = [q["id"] for q in qs if q["group"] in SENSITIVITY]
        a2, b2, c2, d2 = paired(by, sids, "C", other)
        out["sensitivity_groups_1_3_4"] = {"descriptive only": True, "questions": len(sids),
                                           "pairs": {"both right": a2, "C only": b2, f"{other} only": c2,
                                                     "both wrong": d2}}
    if "A" in arms:
        g5 = [q["id"] for q in qs if q["group"] in ("5a", "5b")]
        a_right = sum(by[i, "A"]["correct"] for i in g5)
        out["group5_tripwire"] = {arm: {"right": sum(by[i, arm]["correct"] for i in g5), "A": a_right,
                                        "flag": sum(by[i, arm]["correct"] for i in g5) < a_right - 2}
                                  for arm in arms if arm != "A"}
    return out


def main():
    args = sys.argv[1:]
    other = None
    if "--compare" in args:                          # e.g. --compare C,R (default: C vs B if B is given, else R)
        i = args.index("--compare")
        first, other = args[i + 1].split(",")
        assert first == "C", "the primary comparison is C against another arm"
        del args[i:i + 2]
    qpath, outpath = args[:2]
    dirs = dict(x.split("=", 1) for x in args[2:])
    qs = [json.loads(l) for l in open(qpath) if l.strip()]
    qids, qhash = {q["id"] for q in qs}, sha256(qpath)
    if len(qids) != len(qs):
        raise DataError("duplicate ids in the question file")
    problems, runs = [], {}
    for arm, path in dirs.items():
        expected = every4(qs) if arm == "Brep" else [q["id"] for q in qs]
        rows, missing, extra = load_arm(path, arm, qids, qhash, expected)
        runs[arm] = rows
        if missing:
            problems.append(f"{arm}: {len(missing)} missing questions (first: {missing[:3]})")
        if extra:
            problems.append(f"{arm}: {len(extra)} unexpected questions (first: {extra[:3]})")
    if any(q["group"] == "5a" for q in qs):
        wall_canary()
    arms = [a for a in ("A", "B", "C", "R") if a in runs]   # R: inside-4b's RISC-V-emulator-beside arm
    scored = [score_one(q, runs[arm][q["id"]]) for arm in arms for q in qs if q["id"] in runs[arm]]
    if problems:
        summary = {"incomplete": problems, "verdict": "withheld: the data is incomplete or inconsistent"}
    else:
        summary = summarise(qs, scored, arms, runs, other)
        if "Brep" in runs:
            def content(r):
                return json.dumps(((r.get("raw") or {}).get("choices") or [{}])[0].get("message", {}).get("content"))
            same = sum(content(runs["Brep"][i]) == content(runs["B"][i]) for i in every4(qs))
            summary["determinism_B"] = f"{same}/{len(every4(qs))} identical replies"
    json.dump({"summary": summary, "rows": scored}, open(outpath, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    sys.exit(2 if problems else 0)


def selftest():
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    # Newcombe method 10 against independent values (statsmodels' corrected implementation; Codex's local check)
    for cells, want in [((10, 5, 2, 20), (-0.06439261, 0.22111602)), ((80, 0, 0, 80), (-0.01210532, 0.01210532)),
                        ((160, 0, 0, 0), (-0.02344620, 0.02344620))]:
        _, lo, hi = newcombe_paired(*cells)
        check(f"Newcombe {cells}: ({lo:.6f}, {hi:.6f}) vs {want}",
              abs(lo - want[0]) < 2e-6 and abs(hi - want[1]) < 2e-6)
    _, lo, hi = newcombe_paired(158, 1, 1, 0)
    check(f"almost no disagreement -> equivalent ({lo:.3f}, {hi:.3f})", verdict(lo, hi).startswith("practically"))
    _, lo, hi = newcombe_paired(120, 10, 10, 20)
    check(f"12.5% disagreement -> inconclusive ({lo:.3f}, {hi:.3f})", verdict(lo, hi) == "inconclusive")
    _, lo, hi = newcombe_paired(100, 30, 2, 28)
    check(f"clear difference -> computer better ({lo:.3f}, {hi:.3f})", verdict(lo, hi) == "computer better")
    import random
    for name, probs, true in [("10% disagreement", (0.775, 0.075, 0.025, 0.125), 0.05),
                              ("2% disagreement", (0.88, 0.01, 0.01, 0.10), 0.0),
                              ("30% disagreement", (0.45, 0.20, 0.10, 0.25), 0.10)]:
        rng, cover, trials = random.Random(11), 0, 2000
        cum = [sum(probs[:i + 1]) for i in range(4)]
        for _ in range(trials):
            cnt = [0, 0, 0, 0]
            for _ in range(160):
                u = rng.random()
                cnt[next((i for i, x in enumerate(cum) if u < x), 3)] += 1
            _, lo, hi = newcombe_paired(*cnt)
            cover += lo <= true <= hi
        # at least nominal (within simulation noise); above it is conservative, which can only make equivalence
        # harder to claim (the corrected method 10 is known to over-cover when there are few disagreements)
        check(f"coverage, {name}: {cover / trials:.3f} (at least 0.93)", cover / trials >= 0.93)

    # the code-test wall: right, wrong, forged and rigged answers
    tests = [{"call": "count_vowels('Hello World')", "expect": "3"}, {"call": "count_vowels('')", "expect": "0"}]
    for label, src, want in [
        ("right code passes", "def count_vowels(s):\n    return sum(c in 'aeiouAEIOU' for c in s)\n", True),
        ("wrong code fails", "def count_vowels(s):\n    return 3\n", False),
        ("forged pass list + exit 1 fails",
         "import json, os\nprint(json.dumps([True, True]), flush=True)\nos._exit(1)\n", False),
        ("forged result line + exit 0 fails",
         "import json, os, sys\nsys.stdout.write(json.dumps([{'ok': True, 'value': 0}] * 2) + '\\n')\n"
         "sys.stdout.flush()\nos._exit(0)\n", False),
        ("rigged __eq__ fails", "class X(int):\n    def __eq__(self, o):\n        return True\n"
                                "def count_vowels(s):\n    return X(99)\n", False),
        ("endless loop fails", "def count_vowels(s):\n    while True:\n        pass\n", False),
        ("missing function fails", "x = 1\n", False),
        ("fake 'bwrap:' on stderr is the candidate's failure, not a wall error",
         "import sys\nsys.stderr.write('bwrap: injected by candidate\\n')\nraise SystemExit(1)\n", False),
        ("float where an int is expected fails", "def count_vowels(s):\n    return float(sum(c in 'aeiou' "
                                                  "for c in s.lower()))\n", False),
        ("100 MB of output fails without filling memory",
         "import sys\nsys.stdout.write('x' * 100_000_000)\ndef count_vowels(s):\n    return 0\n", False),
    ]:
        try:
            ok, _ = run_tests(src, tests)
            check(f"code wall: {label}", ok == want)
        except WallError as e:
            check(f"code wall: {label} (wall error {e})", False)

    # exact types: True == 1 and False == 0 in Python, but not here (Codex's case: expectations 1 and 0)
    t10 = [{"call": "count_vowels('a')", "expect": "1"}, {"call": "count_vowels('')", "expect": "0"}]
    ok, detail = run_tests("def count_vowels(s):\n    return bool(s)\n", t10)
    check(f"code wall: bool where 1 and 0 are expected fails ({detail})", not ok)
    ok, detail = run_tests("def count_vowels(s):\n    return sum(c in 'aeiou' for c in s)\n", t10)
    check("code wall: the same tests pass with real ints", ok)
    ok, detail = run_tests("import sys\nsys.stdout.write('x' * 100_000_000)\ndef count_vowels(s):\n    return 0\n", t10)
    check(f"code wall: 100 MB of output stopped by the cap ({detail})", detail == "output too large")

    # classifier on made-up rows
    q = {"id": "t", "group": "1", "answer": "121932631112635269"}

    def row(content, finish="stop", calls=(), http=200, events=(), arm="C", err=None, open_=()):
        return {"arm": arm, "http": http, "seconds": 1.0, "harness_error": err, "tool_calls": list(calls),
                "tool_calls_open": list(open_),
                "raw": {"choices": [{"finish_reason": finish, "message": {"content": content,
                                                                         "strata_mcp": list(events)}}]}}
    good = {"expression": "123456789*987654321", "result": "121932631112635269", "error": None}
    right = "ANSWER: 121932631112635269"
    oom = row(None, finish=None, http=400)
    oom["raw"] = {"error": "verify: instantiate: out of memory"}
    cases = [
        ("right", row(right, calls=[good]), True, None),
        ("right, no tool", row(right, arm="A"), True, None),
        ("wrong, no tool", row("ANSWER: 5", arm="A"), False, "reasoning"),
        ("wrong after a good tool result", row("ANSWER: 5", calls=[good]), False, "final answer"),
        ("cut off", row(right, finish="length", calls=[good]), False, "budget"),
        ("right text but round limit hit", row(right, calls=[good], events=[{"event": "limit"}]), False, "budget"),
        ("right text but a call never ended", row(right, calls=[good], open_=[{"rpc_id": 9}]), False, "budget"),
        ("right text but a trace mismatch", row(right, calls=[good, {"expression": "1+1", "error": "x",
                                                                     "error_kind": "trace_mismatch"}]),
         False, "backend"),
        ("loop with right answer: right, flagged", row(right, calls=[good] * 3), True, None),
        ("refused call then wrong", row("ANSWER: 1", calls=[{"expression": "x" * 300, "result": None,
                                                             "error": "refused: too long", "error_kind": "refused"}]),
         False, "call format"),
        ("refused call then right: recovered", row(right, calls=[{"expression": "x", "error": "refused: x",
                                                                  "error_kind": "refused"}, good]), True, None),
        ("computer budget error then wrong", row("ANSWER: 1", calls=[{"expression": "2**999", "error": "x",
                                                                      "error_kind": "budget"}]), False, "backend"),
        ("computer says error, Python a value", row("ANSWER: 1", calls=[{"expression": "42-848/80/(3/6)-5",
                                                                         "result": "error (too complex: split it)",
                                                                         "error": None, "python_reference": "79/5"}]),
         False, "backend"),
        ("both say error: a bad expression", row("ANSWER: 1", calls=[{"expression": "1+*2", "result": "error",
                                                                      "error": None, "python_reference": "error"}]),
         False, "call format"),
        ("error result with no reference: unknown", row("ANSWER: 1", calls=[{"expression": "1+*2", "result": "error",
                                                                             "error": None}]), False, "unknown"),
        ("HTTP 400 out of memory: backend", oom, False, "backend"),
        ("http 503: transport", row(None, finish=None, http=503), False, "transport"),
        ("timeout: budget", row(None, finish=None, http="timeout"), False, "budget"),
        ("harness", row(None, finish=None, http=None, err="KeyError"), False, "harness"),
    ]
    for label, r, want_ok, want_layer in cases:
        try:
            s = score_one(q, r)
            check(f"classifier: {label} -> {s['layer'] or 'correct'}",
                  s["correct"] == want_ok and s["layer"] == want_layer)
        except Exception as e:  # noqa: BLE001
            check(f"classifier: {label} raised {e!r}", False)
    s = score_one(q, row(right, calls=[{"expression": "x", "error": "refused: x", "error_kind": "refused"}, good]))
    check("recovered faults are listed", s["recovered"] == ["refused"])
    s = score_one(q, row(right, calls=[good] * 3))
    check("a loop is flagged", s["loop"])

    # chains: steps seen in order, not the furthest value
    qc = {"id": "c", "group": "4", "answer": "30", "steps": 3, "intermediates": ["10", "20", "30"]}
    for label, results, want in [("all in order", ["10", "20", "30"], (3, 3)), ("only the last", ["30"], (0, 1)),
                                 ("batched middle", ["10", "30"], (1, 2))]:
        s = score_one(qc, row("ANSWER: 30", calls=[{"expression": "e", "result": v, "error": None} for v in results]))
        check(f"chain: {label} -> in order {s['chain_in_order']}, seen {s['chain_seen_any_order']}",
              (s["chain_in_order"], s["chain_seen_any_order"]) == want)
    s = score_one(qc, row("ANSWER: 30", arm="A"))
    check("chain: no chain measure for arm A", "chain_in_order" not in s)

    # data integrity: duplicates, wrong arm, unknown ids, another question file, missing questions
    with tempfile.TemporaryDirectory() as t:
        def arm_dir(rows, sha=None):
            p = tempfile.mkdtemp(dir=t)
            with open(os.path.join(p, "replies.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            if sha is not None:
                open(os.path.join(p, "questions.sha256"), "w").write((sha + "  q.jsonl\n") if sha else "")
            return p
        F = "f" * 64
        for label, rows, sha in [("duplicate id", [{"id": "x", "arm": "B"}, {"id": "x", "arm": "B"}], F),
                                 ("wrong arm", [{"id": "x", "arm": "C"}], F),
                                 ("unknown id", [{"id": "zzz", "arm": "B"}], F),
                                 ("other question file", [{"id": "x", "arm": "B"}], "0" * 64),
                                 ("no question fingerprint", [{"id": "x", "arm": "B"}], None),
                                 ("empty question fingerprint", [{"id": "x", "arm": "B"}], "")]:
            try:
                load_arm(arm_dir(rows, sha), "B", {"x"}, "f" * 64, ["x"])
                check(f"integrity: {label} rejected", False)
            except DataError:
                check(f"integrity: {label} rejected", True)
        _, missing, _ = load_arm(arm_dir([{"id": "x", "arm": "B"}], F), "B", {"x", "y"}, F, ["x", "y"])
        check("integrity: a missing question is reported", missing == ["y"])
    # summaries: stage 5's C vs B keys are unchanged; C vs R gets its own names
    qs3 = [{"id": "p1", "group": "1", "answer": "4"}, {"id": "p2", "group": "1", "answer": "4"}]

    def srow(i, arm, ok):
        return score_one(qs3[0] if i == "p1" else qs3[1], row("ANSWER: 4" if ok else "ANSWER: 5", arm=arm,
                                                           calls=[good] if arm != "A" else []))
    scored = [srow("p1", "B", True), srow("p2", "B", False), srow("p1", "C", True), srow("p2", "C", True)]
    for r in scored:
        r["id"] = r["id"]
    sc = [dict(r, id=i) for r, i in zip(scored, ["p1", "p2", "p1", "p2"])]
    out = summarise(qs3, sc, ["B", "C"], {"B": {}, "C": {}})
    check("summary C vs B keeps stage 5's keys", "C_minus_B" in out["primary"] and out["primary"]["compared"] == "C vs B"
          and out["primary"]["pairs"]["B only"] == 0 and out["primary"]["pairs"]["C only"] == 1)
    sc2 = [dict(r, arm="R") if r["arm"] == "B" else r for r in sc]
    out2 = summarise(qs3, sc2, ["C", "R"], {"C": {}, "R": {}})
    check("summary C vs R names R", "C_minus_R" in out2["primary"] and out2["primary"]["compared"] == "C vs R"
          and "R only" in out2["primary"]["pairs"])
    out3 = summarise(qs3, sc2 + [dict(r, arm="B") for r in sc2 if r["arm"] == "R"], ["B", "C", "R"],
                     {"B": {}, "C": {}, "R": {}}, other="R")
    check("an explicit comparator wins over the default", out3["primary"]["compared"] == "C vs R")
    print("SELFTEST", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    selftest() if sys.argv[1:2] == ["--selftest"] else main()
