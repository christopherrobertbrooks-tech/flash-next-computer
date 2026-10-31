"""Does flash-linear-attention (fla) run on the V100, and what does it change? Run the same script twice: once as is
(transformers' PyTorch fallbacks) and once with PYTHONPATH pointing at the separate fla folder.

  1. which gated-delta kernels transformers picked;
  2. plain Qwen3.5-4B (bf16 HF): prompt reading (one 4,096-token forward) and greedy decoding (256 new tokens, one
     token per step with a KV cache), timed;
  3. greedy replies (thinking off, 512 tokens) to 5 stage-5 pilot prompts, saved for comparison between the two runs;
  4. the circuit's teacher-forced prefill (upstream's hf path) on a calculator trace: time, and exact on every position
     (upstream's hf.check).

usage (riscv dir on PYTHONPATH, V100 visible, inside the V100 lock): python fla_probe.py BASE_MODEL_DIR PILOT.jsonl OUT.json
"""
import json, sys, time

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

from chipformer.q35.fast import exact_matmuls
from chipformer.rv import calc, core, hf


def sync():
    torch.cuda.synchronize()


def main():
    model_dir, pilot, out = sys.argv[1:4]
    exact_matmuls()
    res = {"fla": is_flash_linear_attention_available(), "causal_conv1d": is_causal_conv1d_available()}
    print("fla available:", res["fla"], "| causal_conv1d available:", res["causal_conv1d"], flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    attn = model.model.layers[0].linear_attn
    res["chunk_kernel"] = getattr(attn, "chunk_gated_delta_rule", None).__module__ if hasattr(attn, "chunk_gated_delta_rule") else None
    print("chunk kernel from:", res["chunk_kernel"], flush=True)
    try:
        with torch.no_grad():
            ids = torch.randint(1000, 100000, (1, 4096), device="cuda")
            model(input_ids=ids[:, :64])
            sync()
            t = time.time()
            model(input_ids=ids)
            sync()
            res["prefill_tokens_per_s"] = round(4096 / (time.time() - t))
            p = tok.apply_chat_template([{"role": "user", "content": "Write a short story about a lighthouse."}],
                                        add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
                                        return_dict=True)["input_ids"].cuda()
            o = model(input_ids=p, use_cache=True)
            cache, nxt = o.past_key_values, int(torch.argmax(o.logits[0, -1]))
            sync()
            t = time.time()
            for _ in range(256):
                o = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
                cache, nxt = o.past_key_values, int(torch.argmax(o.logits[0, -1]))
            sync()
            res["decode_tokens_per_s"] = round(256 / (time.time() - t), 1)
        print("prefill", res["prefill_tokens_per_s"], "tokens/s | decode", res["decode_tokens_per_s"], "tokens/s", flush=True)
        replies = []
        for line in list(open(pilot))[:5]:
            q = json.loads(line)
            p = tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], add_generation_prompt=True,
                                        enable_thinking=False, return_tensors="pt", return_dict=True)
            with torch.no_grad():
                g = model.generate(**{k: v.cuda() for k, v in p.items()}, max_new_tokens=512, do_sample=False)
            replies.append({"id": q["id"], "text": tok.decode(g[0][p["input_ids"].shape[1]:], skip_special_tokens=False)})
        res["replies"] = replies
        print("replies: done", flush=True)
    except Exception as e:  # noqa: BLE001
        res["model_error"] = f"{type(e).__name__}: {str(e)[:400]}"
        print("MODEL PATH FAILED:", res["model_error"], flush=True)
    del model
    torch.cuda.empty_cache()
    try:
        c = core.build()
        expr = "(2**64 - 1) // 7"
        core.snapshot(c, calc.machine(expr))
        toks = calc.machine(expr).run(10**6)
        m, lay, ids = hf.build_model(c, toks, torch.bfloat16)
        m = m.to("cuda")
        x = torch.tensor([[ids[t] for t in toks]], device="cuda")
        with torch.no_grad():
            o = m(input_ids=torch.tensor([[ids[None]]], device="cuda"), use_cache=True)
            cache = o.past_key_values
            hf.inject(c, cache)
            sync()
            t = time.time()
            m(input_ids=x, past_key_values=cache, use_cache=True)
            sync()
            dt = time.time() - t
        exact = hf.check(c, m, lay, ids, toks, "prefill")
        res["circuit_prefill"] = {"tokens": len(toks), "tokens_per_s": round(len(toks) / dt), "exact_positions": exact}
        print("circuit prefill:", res["circuit_prefill"], flush=True)
    except AssertionError as e:
        res["circuit_prefill"] = {"error": "NOT EXACT: " + str(e)[:300]}
        print("circuit prefill NOT EXACT:", str(e)[:200], flush=True)
    except Exception as e:  # noqa: BLE001
        res["circuit_prefill"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print("circuit prefill FAILED:", res["circuit_prefill"]["error"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
