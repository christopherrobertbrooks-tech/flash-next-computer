"""LLAMACPP-PLAN v2 stage 0b: are the executor-built prompt rows equal to the rows the full-checkpoint prefill feeds
Qwen's first layer? Elementwise, in bf16, every prompt position of every fixture. If every row is equal, Qwen's
layers get identical input, so the first-token logits are identical too (the same function on the same input). Any
difference fails the candidate; it prints where (position, dims, sizes). CPU only. Exit 1 on any difference.

usage: python check_prompt_rows.py FIXTURE_DIR
"""
import json, sys
from pathlib import Path

import torch


def main():
    d = Path(sys.argv[1])
    man = json.loads((d / "manifest.json").read_text())
    bad = 0
    for f in man["fixtures"]:
        fx = torch.load(d / f"{f['id']}.pt", weights_only=False)
        hf, ex = fx["prefill_rows"], fx["executor_prompt_rows"].to(hf.dtype)
        assert hf.shape == ex.shape == (len(fx["prompt_ids"]), hf.shape[1]), (hf.shape, ex.shape)
        ne = (hf != ex)
        rows = ne.any(1).nonzero().flatten().tolist()
        dims = sorted(set(ne.nonzero()[:, 1].tolist()))
        line = f"{f['id']:16} {hf.shape[0]} prompt rows: "
        if rows:
            bad += 1
            dd = (hf.float() - ex.float()).abs()
            line += (f"DIFFER at {len(rows)} positions (first {rows[:5]}), dims {dims[:10]}{'...' if len(dims) > 10 else ''}, "
                     f"max |diff| {dd.max().item():.3g}")
        else:
            line += "equal"
        print(line)
    print("PROMPT ROWS", "EQUAL on every fixture" if not bad else f"DIFFER on {bad} fixtures")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
