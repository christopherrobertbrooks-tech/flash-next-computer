# llama.cpp port, stage 2: the split runner (2026-10-31; pilot 1:48-1:59 pm, diagnostics to 2:14 pm)

**Answer:** Qwenputer runs with its Qwen half in llama.cpp and its computer on upstream's fp32 executor, with no HF
checkpoint loaded. On the 12-question pilot every answer ended EOS with **0 lock errors** (188,945 checked chip
tokens). Text ran at **43.5 tokens/s** vs 30 in inside-4b's C on the same questions; the computer at the same ~1,700
tokens/s. A bf16 output head (a bit-exact copy of the source values) is 10.6% faster, picks the same token at all
14,745 fixture positions, and passes the stage-0 gate. Not yet reviewed by Codex.

## What was built
- `llamacpp/split_chat.py`: `SplitChat` (upstream's RvChat on the executor path without the 18.8 GB checkpoint; the
  `_exec_hc` rows from the checkpoint's own embedding values) and `split_generate` (upstream's generate copied line
  for line, the prompt prefill and the Qwen steps on the helper, marked SPLIT; check_head not available).
- `llamacpp/run_split.py`: run_qwenputer's rows (arm S), fail-closed on helper errors, memory sampled the way Codex
  specified (both processes summed per nvidia-smi sample, every 0.5 s).
- `viewer/rvlive.py --backend llama`: the live viewer on the split path.

## Controls
| | Result |
|---|---|
| CPU (`test_split_chat.py`, 13) | split_generate = upstream generate on scripted fakes with a real reference stream (one call, two calls, text budget, computer budget, a wrong chip token, the circuit giving up); reset, prefill at 0..P-1, one row per text step at P, P+1, ..., no chip rows; a circuit-owned token wins. A mutation (the helper's choice over the circuit's) fails 5 controls |
| GPU (`test_split_gpu.py`, 25) | on all 12 fixtures: the executor-built prompt rows = the HF prefill rows; the executor state after them = ExecRow.start's; a free run ends EOS with no lock error, and every row sent equals the fixture's until the first generated difference, which on all 9 diverging answers is at an exact tie (HF margin 0.000); A, B, ..., A reproduces A |
| viewer | 38 live + 33 phase-1 controls still pass |

## Pilot (12 questions, `results/llamacpp/stage2-pilot/`)
| | split (S) | inside-4b C, same 12 |
|---|---|---|
| EOS / lock errors | 12/12, 0 | 12/12, 0 |
| text tokens/s (text-step time) | **43.5** | ~30 |
| chip tokens/s (chip-step time) | ~1,700 | ~1,700 |
| GPU memory | 19,108 MiB sampled max, both processes (parent 1,690, helper 17,418) | (18.98 GiB PyTorch allocated: not comparable) |

Answers differ from C's where the free run took the other side of a tie, so per-question times aren't comparable;
stage 3's 160 questions decide that.

## Diagnostics (`results/llamacpp/stage2-bench/`)
- **Where ops run (`GGML_SCHED_DEBUG=2`):** two splits per graph; everything on CUDA0 except the input-embedding
  GET_ROWS on the CPU (unused with row input). The output MUL_MAT (the tied embedding) runs on CUDA0, for both heads.
- **The pipe's cost:** helper row steps 22.02 ms median (45.4/s) vs llama-bench tg128 45.5 tokens/s (21.98 ms) on the
  same GGUF: the client and pipe add about 0.04 ms a step. The rest of a text step (~1 ms: the executor step, the
  row, ownership) is the computer side.
- **Why 43.5, not plain Qwen's 73:** the widened layers and the f32 head make each token read ~14 GB of weights vs
  ~8 GB for plain Qwen f16; llama-bench gives 45.5 tokens/s for this GGUF, so the split runs at llama.cpp's own speed.
- **bf16 head variant** (`qwen-half-f16-bf16head.gguf`, sha256 `643e78a9...`, 13.05 GB): its token_embd is
  bit-identical to the source bf16 embedding; the other 425 tensors equal the baseline GGUF's. tg128 **50.3** vs 45.5
  tokens/s, helper steps 50.4/s, prompt 2,380 vs 2,154 tokens/s; the same argmax as the f32 head at all 14,745
  fixture positions; the stage-0 gate PASSes with it (99.782% vs control 99.796%).

## For stage 3
Proposal (for review before the run): register the **bf16 head** as stage 3's primary configuration (identical
stored values, identical picks on every fixture position, faster, 1.9 GB smaller), with the f32 head kept as the
stage-0/2 reference. The registered comparison stays C (inside-4b) on the same 160 questions.
