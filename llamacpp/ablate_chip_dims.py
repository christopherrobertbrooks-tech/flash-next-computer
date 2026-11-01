"""Inside-check: does Qwenputer's Qwen half (in llama.cpp) depend on the computer's dims of its input rows? Feed the
stage-0 fixture rows twice through the same helper and GGUF, as recorded and with every dim from 2,560 on set to 0
(the computer's code, tau and mu), teacher-forced (the prompt as one prefill, then one row per text step). Compare
the argmax at every position, and the logit gaps (the top value minus the 2nd..5th) where the top-5 ids agree; tau
adds the same offset to every text logit, so the values shift but the gaps shouldn't.

usage: python ablate_chip_dims.py HELPER GGUF FIXTURE_DIR OUT.json   (V100, in the lock)
"""
import json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from helper_client import Helper  # noqa: E402

D0 = 2560


def run(h, fx, zero):
    P = len(fx["prompt_ids"])
    pre, steps = fx["prefill_rows"].float().clone(), fx["step_rows"].float().clone()
    if zero:
        pre[:, D0:] = 0
        steps[:, D0:] = 0
    h.reset()
    out = [h.rows(list(range(P)), pre.numpy())]
    for k in range(steps.shape[0]):
        out.append(h.rows([P + k], steps[k:k + 1].numpy()))
    return out


def by_owner(fdir, man, res):
    """The same comparison split by who chooses the next token (the stage-0 ownership mask): Qwen-chosen positions
    (the prompt's last and text steps the circuit doesn't own) vs circuit-owned ones (their output is the circuit's,
    deliberately changed by zeroing the code dims)."""
    out = {o: {"positions": 0, "same_argmax": 0, "same_top5_ids": 0, "max_gap_difference_where_top5_same": 0.0}
           for o in ("qwen", "circuit")}
    for f in man["fixtures"]:
        fx = torch.load(fdir / f"{f['id']}.pt", weights_only=False)
        own = ["qwen"] + ["qwen" if s["circuit"] is None else "circuit" for s in fx["steps"]]
        for j, ((ia, va), (ib, vb)) in enumerate(zip(res[f["id"]]["as_recorded"], res[f["id"]]["chip_dims_zeroed"])):
            e = out[own[j]]
            e["positions"] += 1
            e["same_argmax"] += ia[0] == ib[0]
            if ia == ib:
                e["same_top5_ids"] += 1
                d = max(abs((va[0] - x) - (vb[0] - y)) for x, y in zip(va[1:], vb[1:]))
                e["max_gap_difference_where_top5_same"] = max(e["max_gap_difference_where_top5_same"], d)
    return out


def main():
    helper, gguf, fdir, outp = sys.argv[1:5]
    fdir = Path(fdir)
    man = json.loads((fdir / "manifest.json").read_text())
    res, tot, same_arg, same_top5, gap_max, n_cmp = {}, 0, 0, 0, 0.0, 0
    with Helper(helper, gguf, 16384, n_vocab=248320) as h:
        for f in man["fixtures"]:
            fx = torch.load(fdir / f"{f['id']}.pt", weights_only=False)
            a, b = run(h, fx, False), run(h, fx, True)
            res[f["id"]] = {"as_recorded": a, "chip_dims_zeroed": b}
            for (ia, va), (ib, vb) in zip(a, b):
                tot += 1
                same_arg += ia[0] == ib[0]
                if ia == ib:
                    same_top5 += 1
                    ga = [va[0] - v for v in va[1:]]
                    gb = [vb[0] - v for v in vb[1:]]
                    gap_max = max(gap_max, max(abs(x - y) for x, y in zip(ga, gb)))
                    n_cmp += 1
            print(f["id"], len(a), flush=True)
    s = {"all_positions": {"positions": tot, "same_argmax": same_arg, "same_top5_ids": same_top5,
                           "max_gap_difference_where_top5_same": gap_max}, "by_owner": by_owner(fdir, man, res)}
    Path(outp).write_text(json.dumps({"summary": s, "rows": res}) + "\n")
    print(json.dumps(s))


if __name__ == "__main__":
    main()
