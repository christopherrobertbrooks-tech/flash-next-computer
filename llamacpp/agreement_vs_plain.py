"""Inside-check, part 3 (CPU): Qwenputer's HF greedy choice (the token it generated, at Qwen-chosen positions of the
stage-0 fixtures) vs plain Qwen3.5-4B HF bf16's recorded argmax on the same text (plain_hf_reference v2), and vs any
other candidate's helper output (e.g. a llama.cpp run). Prints agreement and the margins at the disagreements.

usage: python agreement_vs_plain.py FIXTURE_DIR OTHER.json [--argmax-field]   (plain HF v2: --argmax-field)
"""
import json, sys
from pathlib import Path

import torch


def main():
    fdir, other = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    use_argmax = "--argmax-field" in sys.argv
    man = json.loads((fdir / "manifest.json").read_text())
    N = S = 0
    margins, per = [], []
    for f in man["fixtures"]:
        fx = torch.load(fdir / f"{f['id']}.pt", weights_only=False)
        gen = fx["generated"]
        top = [fx["prefill_top5"]] + [s["top5"] for s in fx["steps"]]
        nxt = [gen[0]] + [gen[s["gen_index"] + 1] for s in fx["steps"]]
        q = [0] + [k + 1 for k, s in enumerate(fx["steps"]) if s["circuit"] is None]
        pick = lambda j: other[f["id"]][j][2] if use_argmax else other[f["id"]][j][0][0]
        d = [j for j in q if pick(j) != nxt[j]]
        N += len(q)
        S += len(q) - len(d)
        per.append((f["id"], len(q), len(d)))
        margins += [top[j][1][0] - top[j][1][1] for j in d]
    print(f"agreement with Qwenputer's HF choices: {S}/{N} = {100 * S / N:.3f}%")
    print("per question:", per)
    print(f"Qwenputer's margins at the disagreements: max {max(margins) if margins else None}, >= 1.0: "
          f"{sum(m >= 1.0 for m in margins)}, exact ties: {sum(m == 0 for m in margins)}")


if __name__ == "__main__":
    main()
