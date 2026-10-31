"""LLAMACPP-PLAN v2 stage 0d score, with the stop rule fixed in the plan before any result:
- positions: every position in the 12 fixed questions where Qwen chooses the next token (the prompt's last, and each
  text step where the circuit doesn't own the next token); the control uses the same positions;
- split agreement: llama.cpp split top-1 == the fixtures' HF top-1; control: plain llama.cpp f16 top-1 == plain HF
  bf16 top-1;
- STOP if split pooled agreement < control pooled agreement - 1.0 point, or if more than 2 of the 12 questions have a
  split disagreement where HF's margin (top-1 minus top-2 logit) is at least 1.0.
The optional extra question is excluded from the rule. Prints per-question rows; exit 1 on STOP.

usage: python score_premise.py FIXTURE_DIR SPLIT.json PLAIN_HF.json PLAIN_LLAMA.json
"""
import json, sys
from pathlib import Path

import torch


def main():
    d = Path(sys.argv[1])
    split, phf, pll = (json.loads(Path(p).read_text()) for p in sys.argv[2:5])
    man = json.loads((d / "manifest.json").read_text())
    S = C = N = 0
    real_q = 0
    rows = []
    for f in man["fixtures"]:
        if f["id"].startswith("extra"):
            continue
        fx = torch.load(d / f"{f['id']}.pt", weights_only=False)
        hf = [fx["prefill_top5"]] + [s["top5"] for s in fx["steps"]]
        qpos = [0] + [k + 1 for k, s in enumerate(fx["steps"]) if s["circuit"] is None]
        assert len(split[f["id"]]) == len(hf) == len(phf[f["id"]]) == len(pll[f["id"]])
        s_dis = [i for i in qpos if split[f["id"]][i][0][0] != hf[i][0][0]]
        c_dis = [i for i in qpos if pll[f["id"]][i][0][0] != phf[f["id"]][i][0][0]]
        margin = lambda i: hf[i][1][0] - hf[i][1][1]
        real = [i for i in s_dis if margin(i) >= 1.0]
        real_q += bool(real)
        S += len(qpos) - len(s_dis)
        C += len(qpos) - len(c_dis)
        N += len(qpos)
        rows.append(f"{f['id']:14} {len(qpos):5} Qwen positions | split disagree {len(s_dis):3} "
                    f"(first {s_dis[0] if s_dis else '-'}, margin {margin(s_dis[0]) if s_dis else 0:.2f}; "
                    f">=1.0: {len(real)}) | control disagree {len(c_dis):3}")
    print("\n".join(rows))
    sa, ca = 100 * S / N, 100 * C / N
    stop = sa < ca - 1.0 or real_q > 2
    print(f"pooled over {N} positions: split {sa:.2f}%  control {ca:.2f}%  (rule: split >= control - 1.0); "
          f"questions with a split disagreement at margin >= 1.0: {real_q} (rule: <= 2)")
    print("PREMISE", "STOP" if stop else "PASS")
    sys.exit(1 if stop else 0)


if __name__ == "__main__":
    main()
