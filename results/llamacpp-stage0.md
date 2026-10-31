# llama.cpp port, stage 0 (premise): PASS (2026-10-31, 8:40-9:31 am)

Plan: `LLAMACPP-PLAN.md` v2 (eff9991), stop rule fixed before any result. Raw data: `results/llamacpp/stage0/`;
fixtures (202 MB) on the gateway at `/mnt/nvme/flashnext-test/llamacpp/fixtures-v1` (hashes in
`fixtures-manifest.json`). Not yet reviewed by Codex.

| Step | Result |
|---|---|
| 0a fixtures | 12 questions (first 3 per group) through upstream's generate (executor, lock): all EOS, 0 lock errors, 229,750 checked chip tokens; 1 to 9 calls each, so the optional two-call question wasn't needed |
| 0b prompt rows | the executor-built prompt rows equal the full-checkpoint prefill rows **bit for bit** on all 12 prompts (5,753 rows); a one-value mutation is caught (exit 1) |
| 0c cut | layers 11-42 renamed 0-31, final norm, embedding sliced to 248,320 rows: 426 tensors, all bit-identical to the source; tokenizers agree on every text id (the only difference is 250 unused `<\|reserved_N\|>` slots Qwenputer names and the base leaves empty) |
| 0c convert | `convert_hf_to_gguf.py --outtype f32 --no-mtp`, then `llama-quantize F16 --token-embedding-type f32`: 15.0 GB GGUF (sha256 `b0f08db8...`) |
| 0d split vs HF | top-1 agreement **99.79%** over 13,746 Qwen-chosen positions; every disagreement is a near-tie (largest HF margin 0.125 logit) |
| 0d control | plain Qwen3.5-4B, llama.cpp f16 vs HF bf16, same positions: **99.76%** |
| reset | A, B, ..., A reproduces A bitwise (both GGUFs) |
| stop rule | split >= control - 1.0 point: yes; questions with a disagreement at margin >= 1.0: 0 (limit 2). **PASS**; a mutated split (3 high-margin flips) gives STOP |

Notes:
- **f16 range.** The 32 Qwen layers fit f16. The embedding (it is also the output head) and the final norm hold
  values up to ~1e12 and ~1e18 in the computer's columns, beyond f16's 65,504. So the embedding is kept f32;
  llama.cpp stores norms in f32 anyway. The input rows are small (max 0.14; the computer's 22 nonzero dims about 1e-6).
- **The converter** asserted on a missing MTP head; `--no-mtp` is its own option for that. Its "Unknown RoPE type:
  default" warning also appears in the plain model's conversion.
- **Positions:** for embedding input llama.cpp takes four rope sections per row; the helper sends [p, p, p, 0], which
  is llama.cpp's own rule for text tokens and equals Qwenputer's (three rope sections = p).
- **What this does and doesn't show.** Teacher-forced agreement on these 12 answers is a premise check. It doesn't
  show the free-running answers will match; stages 2 and 3 decide that. The 12 questions are also part of stage 3's
  160.
