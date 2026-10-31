"""LLAMACPP-PLAN v2 stage 0c: cut Qwen's half out of the Qwenputer checkpoint as a plain HF checkpoint for llama.cpp's
converter. Layers 11-42 become 0-31 (their weights, norm compensation included, copied bit for bit); the final norm;
the tied embedding sliced to the 248,320 text rows (it is also the output head). The width stays 3,968 and every other
config field is kept; layer_types is sliced. The tokenizer files come from the base Qwen3.5-4B, after checking that its
vocabulary equals Qwenputer's on every text id. Streams one tensor at a time (16 GB host RAM). Refuses to overwrite.

usage (the qwenputer venv): python cut_qwen_half.py QWENPUTER_DIR BASE_DIR OUT_DIR
"""
import json, os, shutil, sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 2 << 30


def main():
    src, base, out = map(Path, sys.argv[1:4])
    os.makedirs(out, exist_ok=False)
    meta = json.loads((src / "chipformer.json").read_text())
    n_chip, v0 = meta["qwen_layers_from"], meta["chip_token_offset"]
    cfg = json.loads((src / "config.json").read_text())
    n = cfg["num_hidden_layers"]

    # the tokenizers must agree on every text id
    qv = json.loads((src / "tokenizer.json").read_text())
    bv = json.loads((base / "tokenizer.json").read_text())
    qmap = {i: t for t, i in qv["model"]["vocab"].items()} | {a["id"]: a["content"] for a in qv["added_tokens"]}
    bmap = {i: t for t, i in bv["model"]["vocab"].items()} | {a["id"]: a["content"] for a in bv["added_tokens"]}
    diff = [i for i in range(v0) if qmap.get(i) != bmap.get(i)]
    assert not diff, f"tokenizers differ on {len(diff)} text ids, first {diff[:5]}"
    assert max(bmap) < v0, (max(bmap), v0)

    cfg2 = dict(cfg, num_hidden_layers=n - n_chip, layer_types=cfg["layer_types"][n_chip:], vocab_size=v0)
    (out / "config.json").write_text(json.dumps(cfg2, indent=1) + "\n")
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        if (base / f).exists():
            shutil.copy(base / f, out / f)

    wmap = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    plan = []                                            # (new name, old name, row slice)
    for k in sorted(wmap):
        if k == "model.embed_tokens.weight":
            plan.append((k, k, v0))
        elif k == "model.norm.weight":
            plan.append((k, k, None))
        elif k.startswith("model.layers."):
            i = int(k.split(".")[2])
            if i >= n_chip:
                plan.append((k.replace(f"model.layers.{i}.", f"model.layers.{i - n_chip}.", 1), k, None))
        else:
            raise SystemExit(f"unexpected tensor {k}")
    shard, size, files, index, total = {}, 0, [], {}, 0

    def flush():
        nonlocal shard, size
        if shard:
            name = f"model-{len(files) + 1:05d}.safetensors"
            save_file(shard, str(out / name), metadata={"format": "pt"})
            files.append(name)
            for k in shard:
                index[k] = name
            shard, size = {}, 0

    handles = {}
    for new, old, rows in plan:
        f = wmap[old]
        if f not in handles:
            handles[f] = safe_open(str(src / f), "pt")
        t = handles[f].get_tensor(old)
        if rows is not None:
            t = t[:rows].clone()
        shard[new] = t
        size += t.numel() * t.element_size()
        total += t.numel() * t.element_size()
        if size >= SHARD_BYTES:
            flush()
    flush()
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": index}, indent=1) + "\n")
    print(f"{len(plan)} tensors, {total / 2**30:.2f} GiB in {len(files)} shards; layers {n_chip}-{n - 1} -> "
          f"0-{n - n_chip - 1}; vocab {v0}")


if __name__ == "__main__":
    main()
