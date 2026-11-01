# llama.cpp port, stage 0 (premise): PASS (2026-10-31, 8:40-9:31 am)

Plan: `LLAMACPP-PLAN.md` v2 (eff9991), stop rule fixed before any result. Raw data: `results/llamacpp/stage0/`;
fixtures (202 MB) on the gateway at `/mnt/nvme/flashnext-test/llamacpp/fixtures-v1` (hashes in
`fixtures-manifest.json`; every artifact's sha256 in `provenance-stage0.json`). Reviewed by Codex
(review/codex-llamacpp-stage0 056631f / 3b7fc7d); its corrections are below.

| Step | Result |
|---|---|
| 0a fixtures | 12 questions (first 3 per group) through upstream's generate (executor, lock): all EOS, 0 lock errors, 195,750 checked chip tokens; 1 to 9 calls each, so the optional two-call question wasn't needed |
| 0b prompt rows | the executor-built prompt rows equal the full-checkpoint prefill rows **bit for bit** on all 12 prompts (5,753 rows); a one-value mutation is caught (exit 1) |
| 0c cut | layers 11-42 renamed 0-31, final norm, embedding sliced to 248,320 rows: 426 tensors, all bit-identical to the source; tokenizers agree on every text id (the only difference is 250 unused `<\|reserved_N\|>` slots Qwenputer names and the base leaves empty) |
| 0c convert | `convert_hf_to_gguf.py --outtype f32 --no-mtp`, then `llama-quantize F16 --token-embedding-type f32`: a **mixed F16/F32** GGUF (248 F16 tensors, 178 F32, the tied embedding/output head F32), 14,314 MiB (15.0 GB on disk), sha256 `b0f08db8...` |
| 0d split vs HF | greedy choice (argmax, lowest id at a tie) agreement **99.782%** over 13,746 Qwen-chosen positions (30 disagreements); every one a near-tie (largest HF margin 0.125 logit) |
| 0d control | plain Qwen3.5-4B, llama.cpp f16 vs HF bf16, same positions: **99.796%** (28) |
| reset | A, B, ..., A reproduces A bitwise (both GGUFs) |
| stop rule | split >= control - 1.0 point: yes (0.014 points below); questions with a disagreement at margin >= 1.0: 0 (limit 2). **PASS** |
| gate | `stage0_gate.py` checks everything before scoring and fails closed: the 12 ids once each, file hashes, EOS and no lock error, row/position/ownership consistency, the prompt-row equality, complete outputs, both reset checks, provenance. Its controls (`test_stage0_gate.py`): a lock error, a bad hash, 1 of 12 questions, a duplicate, a failed reset and a reference without argmax all FAIL; 3 high-margin flips STOP; the real data PASS |

**Corrected after Codex's review.** My first scorer took topk's first id as HF's choice, but topk orders tied values
arbitrarily, while upstream decides by argmax (the lowest id at a tie). That gave 99.79% / 99.76% (kept in
`stage0-summary.txt`). Now the fixtures' HF choice is the token upstream actually generated next, and the plain HF
reference was re-run recording its argmax (`plain-hf-v2.json`; its top-5 lists are identical to the first run's at
all 14,745 positions). So the split is very slightly *below* the control, not above it; the reading is the same.
The capture script also records argmax now, and refuses a fixture with a lock error.

Notes:
- **f16 range.** The 32 Qwen layers fit f16. The embedding (it is also the output head) and the final norm hold
  values up to ~1e12 and ~1e18 in the computer's columns, beyond f16's 65,504. So the embedding is kept f32;
  llama.cpp stores norms in f32 anyway. The input rows are small (max 0.14; the computer's 22 nonzero dims about 1e-6).
- **The computer's columns in the output head (Codex, CPU).** At every Qwen-chosen position the only nonzero computer
  coordinate is tau (2^-40). The 64 residual-writing projections of Qwen's layers have exactly zero rows beyond 2,560
  dims, the final norm scales tau by 2^60, and every text row's tau weight is 2^-40. So tau adds the same offset to
  every text logit: it's **not zero**, but it can't change the argmax. Shrinking the head to 2,560 columns is a
  possible separate optimization with its own exactness check (finite-precision ties), not part of this port.
- **The converter** asserted on a missing MTP head; `--no-mtp` is its own option for that. Its "Unknown RoPE type:
  default" warning also appears in the plain model's conversion.
- **Positions:** for embedding input llama.cpp takes four rope sections per row; the helper sends [p, p, p, 0], which
  is llama.cpp's own rule for text tokens and equals Qwenputer's (three rope sections = p).
- **What this does and doesn't show.** Teacher-forced agreement on these 12 answers is a premise check. It doesn't
  show the free-running answers will match; stages 2 and 3 decide that. The 12 questions are also part of stage 3's
  160.
