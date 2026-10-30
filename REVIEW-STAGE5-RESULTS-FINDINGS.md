# Pass 1 (blind)

Reviewed `stage5` at `a9ad2df` following `REVIEW-STAGE5-RESULTS.md`. Frozen code is annotated tag `stage5-freeze`, commit `1d721dde1d37d608521abd1d538a876c6c4a3f69`. Work is in my own clone on `review/codex-stage5-results`. I did not open the results write-up, README status, guided questions or post-freeze commit messages before this pass was committed and pushed. Hash/date-only history and the frozen commit itself were used for provenance; no post-freeze messages were read.

Review/instruments methods applied. No GPU/model execution, Strata launch, install, gateway mutation or product-branch change. Saved code answers were executed only in the reviewed local CPU bwrap wall. Gateway access was read-only for saved logs/CSVs and current source hashes. Temporary decompression/rescoring artifacts are under `/tmp`.

## Protocol and freeze audit

- Tag target commit time: **2026-10-30 10:42:53 -05:00**; tag creation: **10:45:58**. The final-question addition is committed at **10:46:15**, and the logged run starts at **10:46:26**. These recorded timestamps are consistent with freeze, then final-file commit, then execution.
- `STAGE5-PLAN.md`, `stage5/` code, `computer/` code and scripts have **no difference between the frozen tag and the reviewed results revision**, apart from the added `stage5/final-questions.jsonl`. The actual generator/runner/scorer/backend/parser files currently deployed on the gateway match the frozen/local hashes for all eight checked files. This is a current deployment check, not a cryptographic proof of their bytes at every past launch; there is no per-launch complete immutable runtime image in the supplied artifacts.
- Question file: **180 unique IDs**, SHA256 **`3f97781ee43bccd2641d191afba79b8cd533f826be84585cdfad5e77d5a10bb9`**. Each registered arm/repeat records that hash. Full arms contain exactly the two warm-ups plus all 180 questions, in the registered order. B repeat contains exactly its two warm-ups plus the 40 registered every-fourth primary IDs, in order.
- All request texts match the frozen question prompts, with temperature 0, `max_tokens=8192`, non-streaming, and MCP enabled only in B/C. Base engine arguments are identical: vision off, MTP present, int8 KV, 131072 maximum context, resident experts and 700 MiB reserve. B/C offer the same `math__calc` tool; only backend and run/log paths differ. MCP round cap is 20.
- `run-final.log` records one A, B, C and B-repeat arm in that order, all exit 0: A ends 13:05:44, B 13:56:36, C 15:06:40 and B-repeat 15:19:46. Saved rows have no HTTP failures, harness errors, unfinished calls or >600 s request. The saved evidence does not show replaced primary attempts. It cannot prove that unrecorded attempts never occurred.
- There is also a **post-hoc C diagnostic repeat** (`diag-Crep`, 42 rows including warm-ups), starting 17:18:39. It is separate from the registered primary arms and must not replace their results. Its presence is a diagnostic extension of the protocol, not the planned B repeat.

## Independent scoring

I first used separate review code to parse the final ANSWER values as exact fractions, apply the registered hard-failure checks, tally groups/paired outcomes and compute a scalar corrected Wilson/Newcombe interval. Then I ran the **frozen scorer** over locally decompressed saved arms, using the actual CPU bwrap wall for all 30 code answers. Its full output is structurally identical to committed `scores.json.gz`. All **510 primary-arm numeric-question verdicts** agree with the independent parser/checker. No numerical/scoring finding was identified on these saved records.

| Group | A: alone | B: Python | C: computer |
| --- | ---: | ---: | ---: |
| 1. Exact arithmetic | 22/70 | 68/70 | 68/70 |
| 2. Near limits | 7/10 | 10/10 | 10/10 |
| 3. Word problems | 35/40 | 40/40 | 40/40 |
| 4. Chains | 17/40 | 37/40 | 38/40 |
| **Primary** | **81/160** | **155/160** | **156/160** |
| 5a. Code tests | 8/10 | 9/10 | 9/10 |
| 5b. Code reading | 10/10 | 9/10 | 10/10 |
| **Group 5** | **18/20** | **18/20** | **19/20** |

The code-test failures are reproducible: A fails Roman numeral tests and its nth-prime function times out; B/C fail Roman numeral tests. All other code answers pass the frozen tests. No group-5 regression tripwire triggers. Tool use on group 5 is A 0, B 0, C 5 calls (one question); restraint is distinct from answer correctness.

Paired table for C versus B, over the original 160 primary questions:

| | B correct | B wrong |
| --- | ---: | ---: |
| C correct | 153 | 3 |
| C wrong | 2 | 2 |

Independent difference **+0.625 percentage points**; corrected Newcombe 95% interval **[-2.9829534, +4.4107902] percentage points**. This is wholly inside the registered ±5-point margin: **practically equivalent**, not superior. The descriptive groups-1/3/4 table has 143 both right, 3 C-only, 2 B-only, 2 both wrong (150 questions); it does not add a second primary inference.

Primary median request times: A **58.205 s**, B **11.620 s**, C **18.945 s**. C/B ratio **1.63038**, so C meets the registered “at most 2x and at most 60 s” median rule. This is not a per-question cap: primary requests over 60 s number A 79, B 6 and C 9. Maximum C request is 101.94 s. Sum of all scored request times is A 8311.28 s, B 2995.04 s and C 4142.76 s; these are not identical to whole-arm launch/cleanup elapsed time.

Failure layers from the frozen rule (all 180 scored questions):

| Arm | Layers |
| --- | --- |
| A | budget 75; reasoning 6 |
| B | budget 3; call format 2; final answer 1; reasoning 1 |
| C | budget 3; final answer 1; reasoning 1 |

Recovered-soft-fault questions: B 37, C 43. Loops flagged by the exact repeated-call rule: zero. That zero does **not** mean there was no repetitive model reasoning. Saved A server logs contain 17 reasoning-loop stops; 16 of its 75 length failures end below 8192 tokens (one additional loop stop coincides with 8192). B has one early reasoning-loop stop at 3072. Aggregate completion counts can exceed 8192 after tool rounds (e.g. 82 + 8192), consistent with the registered per-round budget.

Successful/failed call records are retained: B **537** matched begin/end pairs, C **534**, registered B-repeat **122**. All call identities are unique within their instance and all begin/end keys match; embedded per-question call lists exactly reproduce the complete end-record lists, including warm-ups. Each registered tool arm has one recorded tool-server instance. There are **zero recorded trace mismatches**. B has 41 typed refusals; C has 43 refusals and four execution errors. Those C errors are examined below.

## Memory, reference checks and repeatability

Independent recount of gateway `mem.csv` at 500 ms sampling:

| Registered run | Samples | Device maximum MiB | Expert slots |
| --- | ---: | ---: | ---: |
| A | 16,714 | 32,005 | 12,288 |
| B | 6,102 | 32,011 | 12,288 |
| C | 8,406 | 32,467 | 12,288 |
| B repeat | 1,567 | 32,009 | 12,288 |

All peaks/slot counts match the saved summaries. Frozen scoring reports tool-process high-water RSS B 1386 MiB / C 1664 MiB, C throughput **1686.4 circuit tokens/s including reference work**, and scored tool time B 175.6 ms / C 1,160,259.0 ms. Those are the protocol's timing boundaries, not isolated kernel-only performance.

**Important resource qualification:** C has four real `OutOfMemoryError` call records, all on `final-5b-007`. The tool tries increasingly smaller expressions and eventually even `2+37` fails allocating 2 MiB. The model still supplies the correct final value. V3 explicitly permits recovered soft faults, so this question is correctly scored right/recovered rather than changing the registered primary outcome. But zero trace mismatches and a sampled device peak below nominal capacity do not establish an OOM-free or safely margined deployment. The sampled device figure cannot certify instantaneous allocator capacity/fragmentation. No specific allocator cause was isolated by these artifacts.

B repeat: **40/40 identical final reply texts**, all 40 correct. This supports repeatability on that registered subset, not all model state or every output token.

Additional C diagnostic repeat: **32/40 identical final reply texts**, all 40 correct. The eight changed outputs are `final-4-001,005,009,013,021,029,033,037`. Only one changes final correctness: `final-4-029` is wrong in original C (length) and right in the diagnostic. Its 40-case original-C subset is therefore 39/40, versus diagnostic 40/40. This demonstrates that C's full pipeline is not text-repeatable on every sampled chain under the saved circumstances. It does not identify the cause or prove a circuit/reference mismatch. The diagnostic remains post-hoc and cannot be substituted into the primary paired table.

## Interpretation and audit limits

**No numerical corrections are required by the independent rescore.** The registered conclusion is practical accuracy equivalence within the chosen margin, with slower tool/request latency for C. Both tools descriptively improve considerably over A on this synthetic set; no new capability or advantage of C over Python is established.

The write-up must retain three qualifications (to be checked in pass 2):

1. Original-primary and post-hoc-repeat results stay separate; do not promote a best C attempt or claim determinism from B's repeat alone.
2. Four recovered C OOMs remain visible despite the correct final answer and zero trace mismatches; “fits” or “clean run” must not imply error-free memory behavior.
3. Budget layers include early reasoning-loop stops, and the practicality threshold is a median rule. Do not call every A failure an 8192-token exhaustion or every C request under 60 s.

Remaining provenance limitation: the annotated tag, source equality, question commit/hash, saved configs and ordered logs are internally consistent, and current deployed source hashes match. They are not a time-of-launch immutable snapshot proving absence of all unrecorded code changes, trials or hardware variation. The general input inventory predates the final run and is not a new per-launch full-content manifest. State these bounds rather than invent a proven cause for C's repeat differences.

Blind verdict: the saved data support the registered equivalence/practicality verdict **with these resource/repeatability limits**. The extra C repeat is diagnostic evidence, not a replacement primary experiment. No product changes or fresh inference were performed; guided questions and results prose remain unopened until this pass is pushed.
