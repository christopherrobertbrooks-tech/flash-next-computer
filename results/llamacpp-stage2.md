# llama.cpp port, stage 2: the split runner (2026-10-31; pilot 1:48-1:59 pm, diagnostics to 2:14 pm)

**Answer:** Qwenputer runs with its Qwen half in llama.cpp and its computer on upstream's fp32 executor, with no HF
checkpoint loaded. On the 12-question pilot every answer ended EOS with **0 lock errors** (188,945 checked chip
tokens). Text ran at **43.5 tokens/s** vs 30 in inside-4b's C on the same questions; the computer at the same ~1,700
tokens/s. A bf16 output head (a bit-exact copy of the source values) is 10.6% faster, picks the same token at all
14,745 fixture positions, and passes the stage-0 gate. Reviewed by Codex (review/codex-llamacpp-stage2 dae624e /
c23fe76); its corrections are below, and stage 3 is held until they're done.

## What was built
- `llamacpp/split_chat.py`: `SplitChat` (upstream's RvChat on the executor path without the 18.8 GB checkpoint; the
  `_exec_hc` rows from the checkpoint's own embedding values) and `split_generate` (upstream's generate copied line
  for line, the prompt prefill and the Qwen steps on the helper, marked SPLIT; check_head not available).
- `llamacpp/run_split.py`: run_qwenputer's rows (arm S), fail-closed on helper errors, memory sampled for both processes
  per nvidia-smi sample (at the pilot: not yet filtered to the V100's UUID and stopped before the helper's
  shutdown; fixed afterwards, see below).
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
- **Where ops run (`GGML_SCHED_DEBUG=2`):** two splits per logged graph: the input-embedding GET_ROWS (and mapped
  input copies) on the CPU, everything else on CUDA0, including the output MUL_MAT (the tied embedding) for prompt
  and one-token shapes, for both heads. These are logged (and reserved) graphs, not runtime op counts, and "every op on
  the GPU" would be wrong: the input lookup is on the CPU (unused with row input).
- **Helper steps vs llama-bench:** warmed helper row steps average 22.03 ms (median 22.02) after a 447-row prompt;
  the separate token-input llama-bench tg128 reference (depth 0) is about 22.00 ms. These unmatched measurements show
  comparable throughput; they **don't isolate the pipe's overhead** (that would need the same history, context and
  settings with and without the transport). Likewise the ~1 ms between a pilot text step and a helper step is a
  residual across unmatched workloads, not a measured breakdown.
- **Why 43.5, not plain Qwen's 73:** the widened GGUF stores about 14.0 GiB of tensors vs about 8.1 GiB for plain
  Qwen, plausibly contributing to the lower decode throughput (no memory-traffic counters here, so bandwidth isn't
  isolated from kernel shapes or precision). The pilot's text bucket (43.5/s) is close to llama-bench's 45.5 on this
  GGUF, about 4% lower; whole-question time includes much more than that bucket.
- **bf16 head variant** (`qwen-half-f16-bf16head.gguf`, sha256 `643e78a9...`, 13.05 GB): its token_embd is
  bit-identical to the source bf16 embedding; the other 425 tensors equal the baseline GGUF's. tg128 **50.3** vs 45.5
  tokens/s (10.7%), helper steps 50.4/s, prompt 2,380 vs 2,154 tokens/s; 1.835 GiB less stored; the same argmax as the f32 head at all 14,745
  fixture positions; the stage-0 gate PASSes with it (99.782% vs control 99.796%).

## Codex's stage-2 findings and what was done
1. **The free-run check stopped at the first divergence** (and would pass a helper answering EOS at once). Added an
   independent **history oracle**: the split's every sent row along its own history, prefix and suffix, is compared
   with upstream's HF chip-layer path teacher-forced on the same tokens, plus every circuit-owned choice
   (`record_split_history.py`, `oracle_history.py`, `compare_histories.py`, with its own negative controls); and the
   stage-0 gate, run with the exact stage-3 helper and GGUF right before the run, rules out a wrong helper (a new
   control: a helper answering EOS everywhere gets STOP).
2. **Memory scope:** the sampler now filters to the V100's UUID, samples through the helper's shutdown, records each
   sample's time and phase, and is joined before the report (`test_run_split.py`).
3. **Pipe-cost wording:** corrected above.
4. **The scorer dropped arm S:** arm S is scored now, unknown arms and a missing comparator are refused, and inside-4b
   and stage 1 re-score identically (`test_score_arms.py`).
5. **A lock error didn't fail the run:** it now stops the run as INVALID DESIGN after saving the row; any exception
   stops it as incomplete; a prompt that can't fit is refused (`test_run_split.py`).
**History oracle result** (`results/llamacpp/stage2-oracle/`, bf16-head GGUF, the 12 fixture prompts): along the
split's own histories, **all 23,956 rows** (5,753 prompt + 18,203 text) equal the HF chip view's bit for bit, before
and after every divergence from the old fixtures, and **all 189,771 circuit-owned choices** agree; no lock error.
Its three negative controls (a corrupted suffix row, a corrupted prompt row, a changed circuit-owned token) are each
caught. The two upstream computer paths (fp32 executor, bf16 HF modules) agree on these histories.

The bf16 head is registered as stage 3's primary configuration (Codex: sound, chosen on development data) in
`LLAMACPP-PLAN.md` v3, with the full protocol.
