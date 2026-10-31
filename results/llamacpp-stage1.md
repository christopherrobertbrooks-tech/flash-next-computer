# llama.cpp port, stage 1: plain Qwen in llama.cpp with the emulator beside it (2026-10-31, 10:32-11:19 am)

**Answer:** inside-4b's arm R (plain Qwen3.5-4B + the calc firmware on the RV32 reference machine) with the model in
**llama.cpp f16** instead of Hugging Face bf16 scored **129/160** vs R's **130/160**: L minus R = -0.6 points, 95%
interval **-6.7 to +5.5** (inconclusive under the ±5 margin: neither a difference nor equivalence is established). It
was **1.85x faster** at the median (**4.67 vs 8.63 s** per question) and took **2,840 vs 5,491 s** in total; text ran
at **72.7 vs 36.8 tokens/s**. This measures the **combined engine-plus-precision change** (HF bf16 to llama.cpp f16),
not either one alone. Not yet reviewed by Codex.

Run: `llamacpp/run_plain_llama.py` (arm L) at cfbf17f, the f16 GGUF of the base model (sha256 `958a023c...`), helper
`9383c566...`, the same 160 questions (sha256 `e3f98862...`), budgets 8,192 text / 2,000,000 trace tokens. R: inside-4b's
saved run (`results/inside4b-full/full-R`). Scored with `stage5/score.py --compare L,R` (arm L added; inside-4b's
published scores re-score identically with the change). Raw: `results/llamacpp/stage1-full/`.

| | L (llama.cpp f16) | R (HF bf16, inside-4b) |
|---|---|---|
| primary | **129/160** | **130/160** |
| group 1 / 2 / 3 / 4 | 69/70, 10/10, 33/40, 17/40 | 69/70, 10/10, 31/40, 20/40 |
| paired | both right 118, L only 11, R only 12, both wrong 19 | |
| median s per question | **4.67** | **8.63** |
| total s | 2,840 | 5,491 |
| text tokens / tokens per s | 206,509 / 72.7 | 202,043 / 36.8 |
| stops | 154 EOS, 6 text budget | 153 EOS, 7 text budget |
| failure layers (budget, call format, final answer, reasoning) | 6, 6, 13, 6 | 7, 5, 16, 2 |
| GPU memory | 11,074 MiB, helper process (nvidia-smi, CUDA context included) | 8.98 GiB PyTorch peak allocated |

- **Same inputs:** prompt-id hashes identical on 160/160; the same firmware, budgets and emulator code. inside-4b's
  run_plain.py was used as it is on main: it differs from the copy that ran R only by diagnostic fields and the
  computer-budget edge (`>=` vs `>`), which R never approached (largest answer 2.4% of the cap).
- **The texts diverge:** identical full generated text on 81/160 (pilot: 3/12). f16 and bf16 pick differently at
  near-ties, then the calls and lengths change. The accuracy comparison is inconclusive, as inside-4b's was.
- **Memory isn't like for like:** the helper's figure is the whole process on the card (context, KV cache, compute
  buffers); R's is PyTorch's allocated peak.
- **Speed is the robust result:** text tokens per second doubled (72.7 vs 36.8) on similar total text (206k vs 202k
  tokens); the per-question median fell 46%.
- **Pilot first:** 12 questions (the stage-0 set), 12/12 EOS, no errors (`results/llamacpp/stage1-pilot/`).
