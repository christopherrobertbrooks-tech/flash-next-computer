"""EARLYSTOP-PLAN v2: the live check. Compares a live run with the rule switched on (run_split.py --early-stop
--record-decisions, on the registered live-check questions) with the unchanged run's offline evaluation:
- tokens: the live generated ids equal the offline expectation exactly (the unchanged ids up to the refused
  decision's index where the rule stops; all of them otherwise), and the live early_stop flag matches;
- time: each live question's elapsed seconds is at most the offline estimate plus max(2 s, 10%) (the rule's overhead
  and cleanup are what the offline estimate leaves out).
Exit 0 only if every selected question passes; it prints each one.

usage: python earlystop_live_check.py UNCHANGED_RUN_DIR EVAL.json LIVE_RUN_DIR
"""
import json, sys
from pathlib import Path


def check(unchanged_dir, eval_path, live_dir):
    ev = json.loads(Path(eval_path).read_text())
    per = {p["id"]: p for p in ev["per_question"]}
    sel = [i for group in ("rule_stops", "text_budget", "eos_median_and_longest") for i in ev["live_check_ids"][group]]
    base = {json.loads(l)["id"]: json.loads(l) for l in open(Path(unchanged_dir) / "decisions.jsonl")}
    live_d = {json.loads(l)["id"]: json.loads(l) for l in open(Path(live_dir) / "decisions.jsonl")}
    live_r = {json.loads(l)["id"]: json.loads(l) for l in open(Path(live_dir) / "replies.jsonl")}
    results = []
    for qid in sel:
        p = per[qid]
        if qid not in live_d:
            results.append((qid, False, "missing from the live run"))
            continue
        want = base[qid]["tokens"][:p["stop_gen_index"]] if p["rule_stops"] else base[qid]["tokens"]
        got = live_d[qid]["tokens"]
        same = got == want and live_d[qid]["early_stop"] == p["rule_stops"]
        expected = p["seconds_with_rule"]
        limit = expected + max(2.0, 0.10 * expected)
        secs = live_r[qid]["seconds"]
        ok = same and secs <= limit
        why = (f"tokens {'equal' if got == want else 'DIFFER'} ({len(got)} vs {len(want)}), early stop live "
               f"{live_d[qid]['early_stop']} / offline {p['rule_stops']}, {secs:.2f} s vs estimate {expected:.2f} "
               f"(limit {limit:.2f})")
        results.append((qid, ok, why))
    return results, ev["live_check_ids"]["shortfall"]


def main():
    results, shortfall = check(*sys.argv[1:4])
    for qid, ok, why in results:
        print(f"{'ok ' if ok else 'BAD'} {qid}: {why}")
    print("shortfall by category:", shortfall)
    good = results and all(ok for _, ok, _ in results)
    print("LIVE CHECK", "PASSED" if good else "FAILED")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
