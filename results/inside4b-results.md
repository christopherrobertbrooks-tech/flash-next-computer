# Inside-4B results: the same calculator firmware inside the model vs beside it (2026-10-30, 8:12-11:51 pm)

**Answer:** on these 160 maths questions, Qwenputer-4B with its checked circuit executor scored **135/160**, and plain
Qwen3.5-4B with the same calculator firmware on the RV32 reference machine scored **130/160**. The pre-registered
accuracy comparison is **inconclusive**: C minus R = +3.125 points, 95% interval about **-1.55 to +7.97** points. It
crosses 0 and reaches past +5, so neither an advantage nor equivalence is established. C passed the registered
**median-latency** rule at **1.94x** R (16.77 vs 8.63 s; at most 2x and 60 s), and used **2.11x** R's PyTorch peak
allocated GPU memory (18.98 vs 8.98 GiB). C's emitted circuit stream had no reference mismatches. Because model width,
numerics and runtime also differ between the two, **this does not isolate a benefit or a cost of CPU placement**.

Reviewed by Codex (review/codex-inside4b-results 7102005 / 74df78c): every number below reproduces independently (all
320 question/arm decisions, the paired table, the interval, the median rule, every stop). This page carries its
corrections to the interpretation.

Protocol: `INSIDE4B-PLAN.md` v2, frozen at tag `inside4b-freeze-2` (125a82f); questions `inside4b/final-primary.jsonl`
(sha256 `e3f98862...`), the groups 1-4 rows of stage 5's frozen final set. v1 was stopped at R 81/160 after review
(kept, unused). One run per arm, R then C. Raw data: `inside4b-full/` (replies gzipped); scores
`inside4b-full/scores.json.gz`. **The frozen scorer crashed** on R's rows (`KeyError('expression')` in a diagnostic
list, the missing field from Codex's finding 10); scores were computed with a disclosed diagnostic-only patch (a
`call_text` fallback) that doesn't touch answer parsing, hard failures, the statistics or the rules. Codex's
independent regrading of all 320 rows doesn't rely on that patch.

## Results
| Group | C (Qwenputer, executor) | R (plain 4B + reference emulator) |
|---|---|---|
| 1 big-number arithmetic | 69/70 | 69/70 |
| 2 near the computer's limits | 10/10 | 10/10 |
| 3 word problems | 32/40 | 31/40 |
| 4 dependent chains | 24/40 | 20/40 |
| **primary** | **135/160 (84.4%)** | **130/160 (81.3%)** |

Paired: both right 126, C only 9, R only 4, both wrong 21. Without group 2 (descriptive only): 116 / 9 / 4 / 21.

**Failure layers, from the recorded metadata** (not directly comparable between the arms; see below):
| Layer | C | R |
|---|---|---|
| budget (8,192 text tokens) | 4 | 7 |
| call format | 2 | 5 |
| backend | 4 | 0 |
| final answer (tool results fine, answer wrong) | 11 | 16 |
| reasoning (no tool, wrong) | 4 | 2 |

- **C's 4 "backend" failures** (`final-4-001`, `-010`, `-019`, `-024`) are recorded **"error (too long: split it)"**
  capture errors on long chained expressions that the whole-parameter Python check can evaluate. They are not wrong
  circuit instructions or mismatches.
- **The two arms built that Python check differently.** R stored the firmware's capture error itself as its reference
  (so its too-long errors became "call format"); C evaluated its whole parameter text (so its too-long errors became
  "backend"). With C's policy applied to R's saved expressions (Codex, CPU check), three R failures (`final-4-017`,
  `-021`, `-024`) move from call format to backend (R: backend 3, call format 2). No question's right/wrong verdict
  changes. A future runner should record the captured bytes and the intended parameter separately, with one oracle for
  both arms.
- **Recovered questions:** 16 per arm (a firmware error or limit in a call, then a right answer). Calls with a firmware
  error result: R 27, C 28.

## What the two runs share, and where they differ
- **Same inputs:** prompt-id hashes identical on 160/160; the same firmware (sha256 `b643ad61...`); the same budgets.
- **Similar computer work in total, not identical computing:** 1,615,594 trace tokens in C and 1,615,658 in R, over
  423 and 419 calls. Per question, the models often asked for different calculations (only 114 of 160 questions have
  equal trace totals, and equal totals don't prove equal traces); e.g. `final-4-012`: C made no call and R made 12,
  and both were right.
- **C's computer was checked:** every emitted circuit token matched the reference machine (0 lock errors, 0 head
  mismatches). That's agreement on the checked stream on these questions, not a universal proof.
- **The generated texts diverge:** the full generated text (calls and responses included) is identical on 88/160
  questions (group 1: 67/70, group 2: 5/10, group 3: 14/40, group 4: 2/40); the final assistant text on 96/160. All 13
  correctness-discordant pairs are among the questions whose texts diverged.
- **Why they diverge is not identified.** Qwenputer's Qwen part is the base model with norms rescaled for a
  3,968-wide residual (verified element by element by Codex), on a different runtime path; those are plausible
  contributors to different greedy choices, which then change the calls, the token counts, the capture limits and the
  final reasoning. This experiment is not an ablation of any of them, and one run doesn't estimate how repeatable the
  divergence is.

## Time and memory
| | C | R |
|---|---|---|
| median s per question (the registered measure) | 16.77 | 8.63 |
| sum of question times | 7,620.8 s | 5,491.2 s |
| PyTorch peak allocated GPU memory | 18.98 GiB | 8.98 GiB |

The buckets inside C aren't comparable with R's timer:
- C recorded 943.65 s in chip-token steps and 6,438.53 s in text-token steps. Its reference checking and other
  overhead (prefill, setup) fall outside those buckets, about 238.6 s of the rest.
- 1,712 trace tokens/s is trace tokens per chip-step second, without the checker's overhead.
- R recorded 3.77 s around its complete emulator calls (machine setup included).

End-to-end time is the comparable cost.

**Pilot vs full run** (3x in the pilot, 1.94x here): the pilot had 12 primary questions, 3 per group; the full set is
70/10/40/40, so slow near-limit questions fell from 25% to 6% and short arithmetic rose from 25% to 44%. The prompts
also changed, the newline was fixed, and C's computer cap rose from 60,000 to 2,000,000. None of these was isolated, so
the pilot's ratio wasn't a controlled forecast.

## Audit
- **Stops:** EOS 153 (R) and 156 (C); text budget 7 (R) and 4 (C), all at exactly 8,192 tokens. No computer-budget
  stops, lock errors, harness or transport errors. One open call (R `final-4-015`, at the text budget): a hard failure.
- **Budget edge:** the largest answers used 47,127 (R) and 53,161 (C) trace tokens, 2.4% and 2.7% of the 2,000,000
  cap, so the `>` vs `>=` edge was never exercised. Codex's per-row list: `REVIEW-INSIDE4B-TRACE-AUDIT.csv` on its
  review branch.
- **Provenance:** the deployed runners match the frozen tag; reply files match the gateway byte for byte; one recorded
  run per arm (it can't prove no unrecorded run).

## Prediction (written before the run)
"Practically equivalent and not practical." **Equivalence was not established** (inconclusive), and **the predicted
failure of the practicality rule did not occur** (1.94x passed, about 3% under the 2x threshold).

## What this answers
On these questions, putting the computer in the model didn't show a measurable accuracy benefit over the same firmware
on a reference emulator beside the model. The observed +3 points is inside the uncertainty. It cost about 1.9x the
median time and 2.1x the GPU memory. The checked trace is a demonstrated property of this run (on the emitted stream).
"Self-contained inside the weights" is an architectural motivation this executor-plus-checker run doesn't establish.
It isn't a measured result here.
