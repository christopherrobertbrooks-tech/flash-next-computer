"""Does Qwenputer-4B contain the plain Qwen3.5-4B unchanged? Reads tensors one at a time from both checkpoints on the
CPU (no model load, no GPU) and checks, for every base language-model tensor, that Qwenputer's tensor (mapped
`model.language_model.X` -> `model.X`, layer index + qwen_layers_from) holds it unchanged in its top-left block;
for the tensors that differ, whether they're the norms and whether (1 + w_qwenputer) / (1 + w_base) is the constant
sqrt(2560 / 3968) a wider RMSNorm needs (Qwen3.5's norms scale by 1 + w).

usage (qwenputer venv; CPU): python verify_base.py QWENPUTER_DIR BASE_DIR
"""
import collections, json, math, sys

import torch
from safetensors import safe_open


def main():
    Q, B = sys.argv[1:3]
    qi = json.load(open(f"{Q}/model.safetensors.index.json"))["weight_map"]
    bi = json.load(open(f"{B}/model.safetensors.index.json"))["weight_map"]
    off = json.load(open(f"{Q}/chipformer.json"))["qwen_layers_from"]
    handles = {}

    def get(path, idx, name):
        fn = f"{path}/{idx[name]}"
        if fn not in handles:
            handles[fn] = safe_open(fn, "pt")
        return handles[fn].get_tensor(name)

    names = [k for k in bi if k.startswith("model.language_model.")]
    same, diff_kinds, ratios, missing = 0, collections.Counter(), [], []
    for k in names:
        rest = k[len("model.language_model."):]
        if rest.startswith("layers."):
            parts = rest.split(".")
            qk = "model.layers." + str(int(parts[1]) + off) + "." + ".".join(parts[2:])
        else:
            qk = "model." + rest
        if qk not in qi:
            missing.append(qk)
            continue
        tb, tq = get(B, bi, k), get(Q, qi, qk)
        sl = tuple(slice(0, s) for s in tb.shape)
        if tq.dim() == tb.dim() and all(a >= b for a, b in zip(tq.shape, tb.shape)) and torch.equal(tq[sl], tb):
            same += 1
        else:
            diff_kinds[rest.split(".")[-2] if rest.startswith("layers.") else rest] += 1
            r = (1 + tq[sl].float()) / (1 + tb.float())
            ratios.append((r.mean().item(), r.std().item(), r.min().item(), r.max().item()))
        del tb, tq
    want = math.sqrt(2560 / 3968)
    print(f"base language-model tensors: {len(names)}; missing in Qwenputer: {len(missing)}")
    print(f"identical in their base block: {same}; differing: {sum(diff_kinds.values())} {dict(diff_kinds)}")
    if ratios:
        means = [m for m, _, _, _ in ratios]
        print(f"(1+w_q)/(1+w_b) means: {min(means):.5f} .. {max(means):.5f}; largest spread (std) "
              f"{max(s for _, s, _, _ in ratios):.5f}; overall min {min(r[2] for r in ratios):.5f}, "
              f"max {max(r[3] for r in ratios):.5f}; sqrt(2560/3968) = {want:.5f}")
    ok = not missing and set(diff_kinds) <= {"input_layernorm", "post_attention_layernorm", "norm.weight"} \
        and all(abs(m - want) < 2e-3 for m, _, _, _ in ratios)
    print("VERDICT:", "the base model is inside Qwenputer unchanged except the norm rescale" if ok else
          "NOT as claimed: see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
