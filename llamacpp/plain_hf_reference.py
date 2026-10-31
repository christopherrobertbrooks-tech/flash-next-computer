"""LLAMACPP-PLAN v2 stage 0d control, HF side: plain Qwen3.5-4B (bf16, as inside-4b's R arm loads it) teacher-forced on
each fixture's text tokens, the same way the fixtures were made (the prompt as one prefill, then one cached step per
text token). Records the top 5 at every Qwen position: the prompt's last, then after each step's input token.

usage (the qwenputer venv, V100 visible, inside the lock): python plain_hf_reference.py BASE_DIR FIXTURE_DIR OUT.json
"""
import json, sys
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM

from chipformer.q35.fast import exact_matmuls


def top5(logits, v0):
    """[top-5 ids, values, argmax]: the order of tied ids in top-5 is topk's; the greedy choice is the argmax (the
    lowest id at a tie, as torch.argmax and upstream decide)."""
    v, i = torch.topk(logits[:v0].float(), 5)
    return [i.tolist(), v.tolist(), int(torch.argmax(logits[:v0]))]


@torch.no_grad()
def main():
    base, fdir, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    if Path(out).exists():
        sys.exit(f"{out} exists: never overwrite")
    exact_matmuls()
    model = Qwen3_5ForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda").eval()
    man = json.loads((fdir / "manifest.json").read_text())
    res = {}
    for f in man["fixtures"]:
        fx = torch.load(fdir / f"{f['id']}.pt", weights_only=False)
        v0 = man["v0"]
        o = model(input_ids=torch.tensor([fx["prompt_ids"]], device="cuda"), use_cache=True)
        cache, tops = o.past_key_values, [top5(o.logits[0, -1], v0)]
        for s in fx["steps"]:
            o = model(input_ids=torch.tensor([[s["input_id"]]], device="cuda"), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            tops.append(top5(o.logits[0, -1], v0))
        res[f["id"]] = tops
        print(f["id"], len(tops), flush=True)
    Path(out).write_text(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
