# Qwenputer's computer with the stock Qwen in llama.cpp: results (2026-10-31, 11:20 pm to 1:28 am)

**Answer (registered primary, F vs C):** Qwenputer's exact computer (upstream's fp32 executor) with the **stock
Qwen3.5-4B at Q8_0** in llama.cpp (arm **F**) scored **131/160** vs the original Qwenputer's **135/160**: F minus C =
**-2.5 points**, 95% interval **-8.3 to +3.2** (inconclusive under the ±5 margin; a loss greater than five points
stays compatible). It was **2.33x faster at the median** (7.19 vs 16.77 s) and **2.40x in summed time** (3,176 vs
7,621 s). All **1,447,753** emitted chip trace tokens matched the RV32 reference (0 lock errors). One run each.
This is a new system, the exact computer plus a stock, quantized language model, not an exact-weight conversion of
Qwenputer. Not yet reviewed by Codex.

Protocol: `FAST-PLAN.md` v2 (registered before the runs; arms F primary and G secondary, both attempted in the fixed
order F then G).
- **Runtime identities** (`results/fast/fast-artifacts-F.json`, `-G.json`, 110 files each): all matched at startup.
- **Preflight:** each arm's stage-0 gate in token mode PASSed in the same session (F 99.687%, G 99.796% vs control
  99.796%).
- **Status:** both runs ok, 160/160 rows each.
- **Scoring:** `llamacpp/fast_analysis.py score` against the frozen analysis identity `results/fast/fast-analysis.json`
  (12 files verified); both arms passed its validity gate. Raw: `results/fast/`.

## The four registered comparisons
| | role | pairs (both right, first only, second only, both wrong) | difference, 95% interval | verdict | median s | latency rule |
|---|---|---|---|---|---|---|
| **F vs C** | **primary** | 123, 8, 12, 17 | **-2.5, [-8.3, +3.2]** | inconclusive | 7.19 vs 16.77 | holds |
| G vs C | secondary | 120, 9, 15, 16 | -3.75, [-10.0, +2.4] | inconclusive | 7.77 vs 16.77 | holds |
| F vs S | secondary | 121, 10, 10, 19 | 0.0, [-5.75, +5.75] | inconclusive | 7.19 vs 10.00 | holds |
| G vs S | secondary | 129, 0, 2, 29 | -1.25, [-3.9, +1.2] | practically equivalent (within 5 points) | 7.77 vs 10.00 | holds |

The secondary rows are descriptive (no multiplicity correction claimed). G's equivalence with S is not a
confirmatory finding, and G doesn't replace F.

## Per arm
| | F (stock Q8_0) | G (stock f16) | S (stage 3 split) | C (original) |
|---|---|---|---|---|
| primary | 131/160 | 129/160 | 131/160 | 135/160 |
| group 1 / 2 / 3 / 4 | 69, 9, 34, 19 | 69, 10, 33, 17 | 69, 10, 34, 18 | 69, 10, 32, 24 |
| median / summed s | **7.19 / 3,176** | 7.77 / 3,967 | 10.00 / 5,148 | 16.77 / 7,621 |
| text tokens, text-step time, rate | 203,096, 2,274 s, **89.3/s** | 206,509, 3,006 s, 68.7/s | 200,331, 4,177 s, 48.0/s | 182,707, 6,439 s, 28.4/s |
| chip tokens, chip-step time | 1,447,753, 836 s | 1,554,614, 896 s | 1,552,141, 897 s | 1,615,594, 944 s |
| lock errors / harness errors | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| text-budget stops (their summed time) | **8** (805 s) | 6 (765 s) | 4 (701 s) | 4 (1,342 s) |
| GPU memory, sampled max (both processes) | 9,070 MiB | 12,784 MiB | 17,254 MiB | not comparable |

- **Speed:** F is 2.33x C at the median and 1.39x S, and 2.40x C in summed time and 1.62x S. The text-step rate is
  1.86x S's. One run each, with different trajectories; no repeatability measured. The computer is now about 26% of
  F's summed time (836 of 3,176 s).
- **F's budget stops:** 8 of 160 (final-1-060, final-2-003, final-3-015, final-3-027, final-4-001, final-4-007,
  final-4-032, final-4-039), taking 805 s, a quarter of F's total time. The development runs showed this tendency
  (2 of 12 fixtures). The scorer's single "loop" flag in F is a tool-call repetition flag, not a textual-loop
  detector.
- **G = stage 1's plain run with the executor:** G's full generated text equals stage 1's L (plain Qwen f16 with the
  reference emulator beside it) on **160/160** questions. That's expected from the same model, prompts and an exact
  computer either way. So G (129/160) is L's answers with executor-produced, checked computer traces, at a median
  of 7.77 s vs L's 4.67 s (L computes each call in bulk on the CPU; G steps the circuit and checks every token).
- **Memory:** sampled the registered way; F's 9.1 GB is half of S's 17.3 GB. No comparison with C's PyTorch figure.

## What it means
- **The fastest exact-computer system so far:** about 2.3-2.4x the original Qwenputer's speed, with the computer
  exact on every emitted token. The accuracy stays inconclusive against C, like every comparison in this project;
  the observed difference is 4 answers.
- **Quantization's visible cost is length:** F's answers hit the text budget twice as often as C's, and those
  questions are a quarter of its time. Whether that's a real accuracy effect isn't established (F vs S: 0.0 points).
- **Caveats:** one run per arm; engine, weights, precision and process arrangement all differ from C; the 12 fixture
  questions used in development are part of the 160; the bf16 head isn't involved (F and G are stock models).
