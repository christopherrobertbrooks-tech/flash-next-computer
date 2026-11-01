"""EARLYSTOP-PLAN v2, development: candidate early-stop rules evaluated offline on the saved answers of the 160
registered questions (arms F, G, S, C). Development data only; the chosen rule is frozen before fresh questions exist.

The saved replies keep text, not token ids or ownership, so tokens are rebuilt approximately: each reply's full text is
split into the circuit's forced frames (after each </tool_call>: "\n<tool_response>...</think>\n\n", i.e. the frame
and the HALT newline) and the language model's own text, and each part is re-tokenized with the base tokenizer (an
approximation of the greedy ids; the fresh run records the real ones).

Rules:
- budget B: stop once the processed text tokens (the recorded count, forced tokens included) would exceed B;
- repetition (k, r): over the language model's own tokens only, stop at the token that completes the r-th occurrence
  of the same k-token span;
- combined: whichever comes first.
For each rule and arm: premature stops (answers the arm got right that the rule stops), failures stopped (of its
text-budget failures), and the estimated time saved (each stopped question's time scaled by the share of its processed
text after the stop: rough, it can misprice answers with expensive computer calls).

usage: python earlystop_dev.py BASE_TOKENIZER.json REPO_ROOT OUT.json
"""
import gzip, json, re, sys
from pathlib import Path

from tokenizers import Tokenizer

FRAME = re.compile(r"(?<=</tool_call>)(\n<tool_response>.*?</tool_response>\n<think>\n\n</think>\n\n)", re.S)
ARMS = {"F": ("results/fast/F/replies.jsonl.gz", "results/fast/scores/F-vs-C.json.gz"),
        "G": ("results/fast/G/replies.jsonl.gz", "results/fast/scores/G-vs-C.json.gz"),
        "S": ("results/llamacpp/stage3/replies.jsonl.gz", "results/llamacpp/stage3/scores.json.gz"),
        "C": ("results/inside4b-full/full-C/replies.jsonl.gz", "results/fast/scores/F-vs-C.json.gz")}
BUDGETS = [3072, 4096, 5120, 6144]
REPS = [(k, r) for k in (16, 32, 64) for r in (3, 4, 6, 8)]


def segments(tok, text):
    """[(token ids, is the language model's own)] in order."""
    out, pos = [], 0
    for m in FRAME.finditer(text):
        out.append((tok.encode(text[pos:m.start()], add_special_tokens=False).ids, True))
        out.append((tok.encode(m.group(1), add_special_tokens=False).ids, False))
        pos = m.end()
    out.append((tok.encode(text[pos:], add_special_tokens=False).ids, True))
    return out


def rep_stop(segs, k, r):
    """The processed-token index where the repetition rule stops, or None."""
    own, seen, idx = [], {}, 0
    for ids, mine in segs:
        for t in ids:
            if mine:
                own.append(t)
                if len(own) >= k:
                    key = tuple(own[-k:])
                    seen[key] = seen.get(key, 0) + 1
                    if seen[key] >= r:
                        return idx
            idx += 1
    return None


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    root, out = Path(sys.argv[2]), Path(sys.argv[3])
    res = {}
    for arm, (rp, sp) in ARMS.items():
        rows = [json.loads(l) for l in gzip.open(root / rp, "rt")]
        right = {r["id"]: r["correct"] for r in json.load(gzip.open(root / sp, "rt"))["rows"] if r["arm"] == arm}
        recs = []
        for r in rows:
            text = r["raw"]["choices"][0]["message"]["full_text"]
            segs = segments(tok, text)
            n_rebuilt = sum(len(i) for i, _ in segs)
            n = (r.get("inside") or {}).get("text_tokens") or r["run"].get("text_tokens") or n_rebuilt
            recs.append({"id": r["id"], "right": right[r["id"]], "budget_fail": r["run"]["stop"] == "text budget",
                         "seconds": r["seconds"], "n": n, "n_rebuilt": n_rebuilt,
                         "rep": {f"{k},{q}": rep_stop(segs, k, q) for k, q in REPS}})
        res[arm] = recs

    def evaluate(rule):
        row = {}
        for arm, recs in res.items():
            prem = fail = 0
            saved = 0.0
            ids = []
            for x in recs:
                stops = []
                if rule[0] in ("budget", "both") and x["n"] > rule[1]:
                    stops.append(rule[1] / x["n"])
                if rule[0] in ("rep", "both"):
                    s = x["rep"][rule[2]]
                    if s is not None:
                        stops.append(s / max(1, x["n_rebuilt"]))
                if not stops:
                    continue
                f = min(stops)
                prem += x["right"]
                fail += x["budget_fail"]
                saved += x["seconds"] * (1 - f)
                if x["right"]:
                    ids.append(x["id"])
            tot = sum(x["seconds"] for x in recs)
            row[arm] = {"premature": prem, "premature_ids": ids, "failures_stopped": fail,
                        "failures": sum(x["budget_fail"] for x in recs), "saved_s": round(saved, 1),
                        "saved_pct": round(100 * saved / tot, 2)}
        return row

    rules = [("budget", b, None) for b in BUDGETS] + [("rep", None, f"{k},{q}") for k, q in REPS] + \
            [("both", b, f"{k},{q}") for b in BUDGETS for k, q in REPS]
    table = [{"rule": r, **{"by_arm": evaluate(r)}} for r in rules]
    out.write_text(json.dumps({"tables": table, "token_count_check": {
        a: {"recorded": sum(x["n"] for x in v), "rebuilt": sum(x["n_rebuilt"] for x in v)} for a, v in res.items()}},
        indent=1) + "\n")
    for t in table:
        r, b = t["rule"], t["by_arm"]
        name = {"budget": f"budget {r[1]}", "rep": f"rep {r[2]}", "both": f"budget {r[1]} + rep {r[2]}"}[r[0]]
        print(f"{name:26} " + " | ".join(f"{a}: prem {b[a]['premature']}, fails {b[a]['failures_stopped']}/{b[a]['failures']}, "
                                         f"saved {b[a]['saved_pct']}%" for a in "FGSC"))


if __name__ == "__main__":
    main()
