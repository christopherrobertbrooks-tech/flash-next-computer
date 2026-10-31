# Inside-4B results: the same computer inside the model vs beside it (2026-10-30, 8:12-11:51 pm)

**Answer:** on 160 maths questions, Qwenputer-4B with its computer inside got **135** right, and plain Qwen3.5-4B
with the same calculator firmware on the RV32 reference machine beside it got **130**. The pre-registered verdict is
**inconclusive**: +3.1 points, 95% interval -1.6 to +8.0 (the interval isn't entirely inside +/-5, and not entirely
above 0). The inside version took **1.94x** the median time per question (16.8 vs 8.6 s), just inside the
pre-registered 2x limit, so it counts as practical, and **2.1x** the GPU memory (19.0 vs 9.0 GiB peak allocated).
Its computer never disagreed with the reference machine.

Codex's wording (INSIDE4B-PLAN.md v2): on these 160 primary questions, Qwenputer's checked fast circuit executor and
plain Qwen3.5-4B with the same calculator firmware on the RV32 reference machine had the paired result below. The two
also differ in model width, numerics and runtime path, so **this doesn't isolate where the CPU lives**.

Protocol: `INSIDE4B-PLAN.md` v2, frozen at tag `inside4b-freeze-2` (125a82f); questions `inside4b/final-primary.jsonl`
(sha256 `e3f98862...`), the groups 1-4 rows of stage 5's frozen final set. v1 was stopped at R 81/160 after review
(kept, unused). Raw data: `inside4b-full/` (replies gzipped); scores `inside4b-full/scores.json.gz`
(`stage5/score.py --compare C,R`).

## Results
| Group | C inside (Qwenputer) | R beside (plain 4B + emulator) |
|---|---|---|
| 1 big-number arithmetic | 69/70 | 69/70 |
| 2 near the computer's limits | 10/10 | 10/10 |
| 3 word problems | 32/40 | 31/40 |
| 4 dependent chains | 24/40 | 20/40 |
| **primary** | **135/160** | **130/160** |

Paired: both right 126, inside only 9, beside only 4, both wrong 21. Difference +0.031, Newcombe 95% [-0.016,
+0.080] -> **inconclusive**. Without group 2 (descriptive): 116 / 9 / 4 / 21.

| Failure layer | C | R |
|---|---|---|
| budget (8,192 text tokens) | 4 | 7 |
| call format | 2 | 5 |
| backend (the firmware's "too complex" on a long chained expression, where Python gives a value) | 4 | 0 |
| final answer (tool results fine, answer wrong) | 11 | 16 |
| reasoning (no tool, wrong) | 4 | 2 |

Both arms recovered from 16 tool faults (refusals or limits, then a right answer). No loops flagged, no harness
errors, no open calls in C (1 in R, at the text budget).

## Why they differ: the language model's path, not the computing
- **Same prompts:** prompt-id hashes identical on 160/160 (Codex also checked all 160).
- **Same computing work:** 1,615,594 trace tokens in C, 1,615,658 in R, over 423 and 419 calls; every C token
  checked against the reference (0 lock errors, 0 head mismatches).
- **Same replies on 88 of 160 questions** (55%): 67 of 70 in group 1, 5/10, 14/40, and 2 of 40 chains. Where the
  replies are identical, the verdicts are identical. All 13 disagreements come from the other 72, where the two
  models chose different words, steps or calls, e.g. `final-4-012`: inside made no call, beside made 12. Qwenputer's
  Qwen part is the base model with rescaled norms in a 3,968-wide residual (Codex verified every element); that
  changes greedy choices on long generations, which then compound.
- So the 9 vs 4 split mostly measures how two numerically different copies of the same 4B model wander on long
  chains. It isn't evidence that a computer inside the weights computes better: the computing was identical.

## Time and memory
| | C inside | R beside |
|---|---|---|
| median s per question | 16.8 | 8.6 |
| mean s per question | 47.6 | 34.3 |
| whole arm | 2.1 h | 1.5 h |
| time in the computer | 944 s (circuit executor, 1,712 trace tokens/s, reference check included) | 3.8 s (Python emulator) |
| GPU peak allocated | 18.98 GiB | 8.98 GiB |

The computer itself costs C about 15 minutes over the run; the rest of C's extra time is the wider model writing its
text more slowly (6,439 s of text in C).

## Audit (Codex's list)
- **Budgets:** no computer-budget stops; the largest answer used 53,161 trace tokens (C) and 47,127 (R) of
  2,000,000, so the exact-cap edge (finding 9) was never reached. Text-budget stops: C 4, R 7.
- **Mismatches:** none.
- **Scorer:** one post-freeze fix needed to score at all: R's rows have no `expression` field (finding 10), and the
  diagnostic list `computer_error_results` indexed it; it now falls back to `call_text`. It affects no verdict logic.
- **Prediction (written before the run):** practically equivalent and not practical. **Both wrong:** the accuracy
  result is inconclusive (leaning to inside, by path noise), and the time ratio (1.94x) came in under the 2x limit.

## What this answers
Putting the computer inside the model's weights didn't make the answers more correct in a way we can show: the
computing is identical either way, and the difference in right answers comes from two copies of the model taking
different paths. It costs about twice the time and twice the memory. The inside computer's real value is the one
that doesn't show up in a score: it's self-contained and every step is checkable, which the beside emulator also is.
