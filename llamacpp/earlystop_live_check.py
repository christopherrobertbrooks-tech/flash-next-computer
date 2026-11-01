"""EARLYSTOP-PLAN v2: the live check, and the only place the final verdict is written.

It compares a live run with the rule switched on (run_split.py --early-stop --record-decisions, on the registered
live-check questions) with the unchanged run's offline evaluation, after validating **both** runs (Codex early-stop
review, findings 1-3):
- **startup:** status ok (so normal completion and helper shutdown); arm F; token mode; decisions recorded; the
  early-stop setting off (unchanged) or on (live); the registered budgets (8,192 text, 2,000,000 chip, context
  16,384); every artifact verified against the frozen runtime manifest, bound by its sha256 (`expect_sha256`);
- **rows:** no lock error, trace mismatch or harness error;
- **the live questions:** exactly the selected ids, and the run's question file is the evaluation's live-questions
  file (by sha256);
- **decisions:** the evaluator's timestamp and index checks (earlystop_eval.check_decisions), on the live run too.
Then, per selected question:
- **tokens:** the live ids equal the offline expectation exactly (the unchanged ids before the refused decision's
  index where the rule stops, all of them otherwise), and the early-stop flag matches;
- **time:** the live monotonic duration is finite and positive, and at most the offline estimate plus max(2 s, 10%).
The final verdict: "withheld" unless both runs are valid and every selected question passes; otherwise it's the
candidate verdict ("keep" / "don't keep"). It's written to FINAL.json together with the live results.

usage: python earlystop_live_check.py UNCHANGED_RUN EVAL.json LIVE_QUESTIONS.jsonl LIVE_RUN RUNTIME_MANIFEST FINAL.json
"""
import hashlib, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import earlystop_eval as EV  # noqa: E402

BUDGETS = {"max_new": 8192, "max_chip": 2_000_000, "n_ctx": 16384}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_problems(run_dir, manifest, early_stop):
    run_dir = Path(run_dir)
    if not (run_dir / "startup.json").exists():
        return ["no startup.json"]
    info = json.loads((run_dir / "startup.json").read_text())
    man = json.loads(Path(manifest).read_text())
    p = []
    if info.get("status") != "ok":
        p.append(f"status {info.get('status')!r} ({info.get('reason')})")
    if info.get("arm") != "F" or info.get("lm_input") != "tokens" or not info.get("record_decisions"):
        p.append("not arm F in token mode with recorded decisions")
    if bool(info.get("early_stop")) != early_stop:
        p.append(f"early_stop {info.get('early_stop')!r}, expected {early_stop}")
    if any(info.get(k) != v for k, v in BUDGETS.items()):
        p.append(f"budgets {[(k, info.get(k)) for k in BUDGETS]}")
    if info.get("expect_sha256") != sha(manifest) or info.get("artifacts_verified") != len(man["files"]):
        p.append("not run against the frozen runtime manifest")
    for l in open(run_dir / "replies.jsonl"):
        r = json.loads(l)
        if (r.get("inside") or {}).get("lock_error") or r.get("harness_error") or \
                any(c.get("error_kind") == "trace_mismatch" for c in r.get("tool_calls") or []):
            p.append(f"{r['id']}: lock, trace or harness error")
    return p


def check(unchanged_dir, eval_path, live_questions, live_dir, manifest):
    ev = json.loads(Path(eval_path).read_text())
    per = {p["id"]: p for p in ev["per_question"]}
    sel = [i for g in ("rule_stops", "text_budget", "eos_median_and_longest") for i in ev["live_check_ids"][g]]
    probs = [f"unchanged: {x}" for x in run_problems(unchanged_dir, manifest, False)]
    probs += [f"live: {x}" for x in run_problems(live_dir, manifest, True)]
    live_dir = Path(live_dir)
    qs_sha = (live_dir / "questions.sha256").read_text().split()[0] if (live_dir / "questions.sha256").exists() else None
    if qs_sha != sha(live_questions):
        probs.append("live: not run on the evaluation's live-questions file")
    base = {json.loads(l)["id"]: json.loads(l) for l in open(Path(unchanged_dir) / "decisions.jsonl")}
    live_d = {json.loads(l)["id"]: json.loads(l) for l in open(live_dir / "decisions.jsonl")}
    live_r = {json.loads(l)["id"]: json.loads(l) for l in open(live_dir / "replies.jsonl")}
    if sorted(live_r) != sorted(sel) or sorted(live_d) != sorted(sel):
        probs.append(f"live: questions {sorted(live_r)} != selected {sorted(sel)}")
    results = []
    for qid in sel:
        p = per[qid]
        if qid not in live_d or qid not in live_r:
            results.append((qid, False, "missing from the live run"))
            continue
        try:
            EV.check_decisions(qid, live_d[qid], live_r[qid]["run"]["stop"])
        except ValueError as e:
            results.append((qid, False, f"invalid live decisions: {e}"))
            continue
        want = base[qid]["tokens"][:p["stop_gen_index"]] if p["rule_stops"] else base[qid]["tokens"]
        got = live_d[qid]["tokens"]
        same = got == want and live_d[qid]["early_stop"] == p["rule_stops"]
        expected = p["seconds_with_rule"]
        limit = expected + max(2.0, 0.10 * expected)
        secs = live_d[qid]["seconds_monotonic"]
        ok = same and math.isfinite(secs) and 0 < secs <= limit
        results.append((qid, ok, f"tokens {'equal' if got == want else 'DIFFER'} ({len(got)} vs {len(want)}), early "
                                 f"stop live {live_d[qid]['early_stop']} / offline {p['rule_stops']}, {secs:.3f} s vs "
                                 f"estimate {expected:.3f} (limit {limit:.3f})"))
    passed = not probs and bool(results) and all(ok for _, ok, _ in results)
    cand = ev["summary"]["candidate_verdict"]
    final = {"candidate_verdict": cand, "live_check_passed": passed, "validity_problems": probs,
             "results": [{"id": q, "ok": ok, "detail": w} for q, ok, w in results],
             "shortfall": ev["live_check_ids"]["shortfall"],
             "final_verdict": ("withheld (the live check didn't pass)" if not passed else
                               "keep" if cand == "candidate keep" else "don't keep")}
    return final


def main():
    unchanged, ev, lq, live, manifest, out = sys.argv[1:7]
    final = check(unchanged, ev, lq, live, manifest)
    Path(out).write_text(json.dumps(final, indent=1) + "\n")
    for p in final["validity_problems"]:
        print("INVALID", p)
    for r in final["results"]:
        print(f"{'ok ' if r['ok'] else 'BAD'} {r['id']}: {r['detail']}")
    print("shortfall by category:", final["shortfall"])
    print("LIVE CHECK", "PASSED" if final["live_check_passed"] else "FAILED", "| final verdict:", final["final_verdict"])
    sys.exit(0 if final["live_check_passed"] else 1)


if __name__ == "__main__":
    main()
