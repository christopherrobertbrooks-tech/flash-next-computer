# Fused-executor premise check: where the computer's time goes (2026-11-01)

Builds on [Qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal (MIT); please cite it as given in
the [README](../README.md#credits-and-citation).

**Answer:** fusing the executor's kernel launches has almost nothing to win. Inside the CUDA graph, the gaps between
its 42 kernel launches per step add up to about **5.6 us of 517 us** (about 1%). The computer's time is real kernel
work: in arm F's loop a chip token costs about 588 us, of which about 509 us is GPU work and about 80 us Python
bookkeeping. Measurement only: `llamacpp/profile_chip_step.py`, raw `results/fused/profile-chip.json`; arm F (the
stock Qwen Q8_0 in llama.cpp plus upstream's fp32 executor) on two fresh questions (fresh-2-007 with 73,343 chip
tokens, fresh-1-061 with 4,905), V100 at 1,380 MHz. One run; not reviewed by Codex.

| Per chip token (fresh-2-007) | us | share |
|---|---|---|
| the loop's chip bucket | 594.5 | |
| chip step, wall (embedding copy, graph replay, decode) | 588.0 (median) | 100% |
| graph replay, GPU time (CUDA events) | 508.9 (median) | 87% |
| the rest: Python and host work | about 80 | 13% |

| Inside one replay (back to back, 2,000 replays; profiler over 50) | us per step |
|---|---|
| back-to-back replay time | 516.6 |
| sum of kernel durations | 511.1 |
| gaps (launch and scheduling) | 5.6 |
| `_head_kernel` (10 per step) | 254.9 |
| `_ffn_kernel` (10 per step) | 192.9 |
| `_apply_kernel` (20 per step) | 52.5 |
| the embedding index and copy (1 each) | 10.7 |

## Ceilings (each one an upper bound, not a prediction)
- **Fusing launches:** at most about 5.6 us per step (1%).
- **A device-resident decode loop** (code decode and the next embedding on the GPU, several steps per sync): at
  most the about 80 us of host work, about 13% of chip time. Chip time is about 26% of F's question time, so that's
  about 3% of a question.
- **Folding the apply kernels into the head/FFN kernels:** at most their 52.5 us (about 10% of chip time). It's a
  correctness-risky redesign (cross-block atomic accumulation, history shifts: Codex's speed review).
- **Faster head and FFN kernels:** 448 us, 87% of GPU time. Earlier, 64 independent machines ran 10x faster in total
  than one (`results/speed-results.md`), which suggests a batch-1 step leaves most of the GPU idle. More parallelism
  inside these kernels is the only lever with a large ceiling (if the computer ran 2x faster, about 13% of F's
  question time). It's kernel research, with exactness to keep at every step.
