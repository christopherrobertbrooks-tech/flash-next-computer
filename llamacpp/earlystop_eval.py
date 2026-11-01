"""EARLYSTOP-PLAN v2: apply the frozen rule offline to an unchanged run that recorded its decisions, and pick the
live-check questions by the registered procedure.

For each question: the language model's decided tokens (decisions.jsonl, in order) go through
earlystop_rule.stop_index. If the rule refuses decision d, the stopped answer is the generated tokens before d's index
(a failure) and its estimated time is d's recorded time (when that token was decided, on the question's clock).
Otherwise the question is unchanged. The estimate leaves out the rule's own overhead and any cleanup; the live check
measures those.

Correctness of the unchanged answers comes from stage5/score.py on that run (`--scores`). Report: summed and median
time unchanged vs with the rule (estimated), the saving, premature stops (answers the unchanged run got right that
the rule stops; listed), failures stopped, text-budget failures stopped, and the registered verdict (keep if the
summed saving is >= 10% and premature stops <= 1).

Live-check selection (registered): by question id order, the first 2 the rule stops; the first 2 that reach the
8,192-token text budget without a rule stop; and 2 that end at EOS without a rule stop, the one of median text length
and the longest. Categories with fewer are used whole and the shortfall reported.

usage: python earlystop_eval.py RUN_DIR SCORES.json OUT.json
"""
import json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import earlystop_rule as R  # noqa: E402


def evaluate(run_dir, scores_path):
    run_dir = Path(run_dir)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(run_dir / "replies.jsonl")}
    dec = {json.loads(l)["id"]: json.loads(l) for l in open(run_dir / "decisions.jsonl")}
    right = {r["id"]: r["correct"] for r in json.load(open(scores_path))["rows"]}
    assert set(rows) == set(dec) == set(right), "the run, its decisions and its scores cover different questions"
    per = []
    for qid in rows:                                     # the run's own order (the question file's)
        r, d = rows[qid], dec[qid]
        assert not d["early_stop"], f"{qid}: the run already stopped early (not an unchanged run)"
        k = R.stop_index([tok for _, tok, _ in d["decisions"]])
        stop = d["decisions"][k] if k is not None else None
        per.append({"id": qid, "right": right[qid], "stop": r["run"]["stop"], "seconds": r["seconds"],
                    "text_tokens": r["inside"]["text_tokens"], "rule_stops": stop is not None,
                    "stop_gen_index": stop[0] if stop else None, "stop_seconds": stop[2] if stop else None,
                    "seconds_with_rule": stop[2] if stop else r["seconds"],
                    "right_with_rule": right[qid] and stop is None})
    base = [p["seconds"] for p in per]
    rule = [p["seconds_with_rule"] for p in per]
    prem = [p["id"] for p in per if p["right"] and p["rule_stops"]]
    saving = 1 - sum(rule) / sum(base)
    summary = {"questions": len(per), "right_unchanged": sum(p["right"] for p in per),
               "right_with_rule": sum(p["right_with_rule"] for p in per),
               "premature_stops": len(prem), "premature_ids": prem,
               "rule_stops": sum(p["rule_stops"] for p in per),
               "failures_stopped": sum(p["rule_stops"] and not p["right"] for p in per),
               "budget_failures": sum(p["stop"] == "text budget" for p in per),
               "budget_failures_stopped": sum(p["rule_stops"] and p["stop"] == "text budget" for p in per),
               "summed_seconds_unchanged": round(sum(base), 2), "summed_seconds_with_rule_estimated": round(sum(rule), 2),
               "saving_estimated": round(saving, 4),
               "median_unchanged": statistics.median(base), "median_with_rule_estimated": statistics.median(rule),
               "verdict": "keep" if saving >= 0.10 and len(prem) <= 1 else "don't keep"}
    stopped = [p["id"] for p in per if p["rule_stops"]][:2]
    budget = [p["id"] for p in per if not p["rule_stops"] and p["stop"] == "text budget"][:2]
    eos = [p for p in per if not p["rule_stops"] and p["stop"] == "eos"]
    picks = []
    if eos:
        by_len = sorted(eos, key=lambda p: (p["text_tokens"], p["id"]))
        picks = [by_len[(len(by_len) - 1) // 2]["id"], by_len[-1]["id"]]
    live = {"rule_stops": stopped, "text_budget": budget, "eos_median_and_longest": list(dict.fromkeys(picks)),
            "shortfall": {"rule_stops": 2 - len(stopped), "text_budget": 2 - len(budget),
                          "eos": 2 - len(set(picks))}}
    return summary, per, live


def main():
    run_dir, scores, out = sys.argv[1:4]
    summary, per, live = evaluate(run_dir, scores)
    Path(out).write_text(json.dumps({"summary": summary, "live_check_ids": live, "per_question": per}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print("live-check questions:", json.dumps(live))


if __name__ == "__main__":
    main()
