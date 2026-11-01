"""Inside-check, part 1 (CPU): which of the 1,408 computer dims (2,560 on) do the 32 Qwen layers read or write, what
do the norms do there, which computer dims does the output head use, and is mu (the mute marker) ever nonzero in the
rows Qwen received on the stage-0 fixtures? Prints one line per weight kind and the fixture counts.

usage: python inside_check_weights.py QWEN_HALF_DIR QWENPUTER_MODEL_DIR FIXTURE_DIR
"""
import json, sys
from pathlib import Path

import torch
from safetensors import safe_open

D0 = 2560


def main():
    half, model, fdir = map(Path, sys.argv[1:4])
    meta = json.loads((model / "chipformer.json").read_text())
    code, tau, mu = set(meta["code_dims"]), meta["tau_dim"], meta["mu_dim"]
    lab = lambda c: "tau" if c == tau else "mu" if c == mu else "code" if c in code else "other"
    w = json.loads((half / "model.safetensors.index.json").read_text())["weight_map"]
    H = {}

    def g(k):
        if w[k] not in H:
            H[w[k]] = safe_open(str(half / w[k]), "pt")
        return H[w[k]].get_tensor(k).float()

    names = sorted({k.split(".", 3)[3] for k in w if k.startswith("model.layers.")})
    for n in names:
        reads, writes, norm, nl = {}, 0.0, None, 0
        for i in range(32):
            k = f"model.layers.{i}.{n}"
            if k not in w:
                continue
            nl += 1
            t = g(k)
            if t.dim() == 2 and t.shape[1] == 3968:
                for c in (t[:, D0:] != 0).any(0).nonzero().flatten().tolist():
                    reads[D0 + c] = max(reads.get(D0 + c, 0), t[:, D0 + c].abs().max().item())
            if t.dim() == 2 and t.shape[0] == 3968:
                writes = max(writes, t[D0:].abs().max().item())
            if t.dim() == 1 and t.shape[0] == 3968:              # stored w (gain 1 + w): the max over all layers
                norm = max(norm or 0.0, t[D0:].abs().max().item())
        rd = {f"{c}({lab(c)})": round(v, 3) for c, v in sorted(reads.items())}
        print(f"{n:34} {nl:2} layers | reads computer dims {rd or 'none'} | writes max {writes:.3g}"
              + ("" if norm is None else f" | norm stored w on computer dims, max |w| over layers {norm:.3g} (gain 1 + w)"))
    nw = g("model.norm.weight")
    kept = [(D0 + i, lab(D0 + i), (1 + nw[D0 + i]).item()) for i in ((1 + nw[D0:]) != 0).nonzero().flatten().tolist()]
    print("final norm gain (1+w) nonzero on computer dims:", sorted({(l, v) for _, l, v in kept}), f"({len(kept)} dims)")
    man = json.loads((fdir / "manifest.json").read_text())
    rows = mu_on = 0
    for f in man["fixtures"]:
        fx = torch.load(fdir / f"{f['id']}.pt", weights_only=False)
        for r in (fx["prefill_rows"], fx["step_rows"]):
            rows += r.shape[0]
            mu_on += int((r[:, mu].float() != 0).sum())
    print(f"rows Qwen received on the fixtures: {rows}; with mu != 0: {mu_on}")


if __name__ == "__main__":
    main()
