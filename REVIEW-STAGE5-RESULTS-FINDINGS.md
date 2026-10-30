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

# Pass 2 (guided)

Blind pass `ff80127` was successfully pushed before opening `REVIEW-STAGE5-RESULTS-PASS2.md`, `results/stage5-results.md` or README status. The guided pass answers the six questions below. It does not change the registered scores or substitute the diagnostic C repeat.

## 1. Agreement with the write-up and findings

The write-up's core accuracy table, paired table, corrected interval, median practicality rule, recovered counts, normal-behavior totals and sampled memory maxima agree with the independent audit. Its conclusion of no demonstrated practical accuracy advantage over Python is supported for this frozen benchmark. Its resource/repeatability disclosures are materially useful.

The following prose needs correction; no fresh model runs are required to make these edits:

1. **REDLINE — the diagnostic does not establish either a causal state switch or absence of a co-residency effect.** The write-up moves from “the computer's presence … seemed to change” to “the computer's presence doesn't change Flash-Next's output,” then says Strata “switched into a different state” and “something accumulates.” README retains the opposite, stronger old claim that the computer's presence changes long generations. The observed short repeat excludes an **inevitable** effect of simply having a computer process present on every launch; it does not exclude a duration/history/resource-dependent co-residency effect, and no specific state or accumulation variable was measured. **Smallest replacement in both places:** “The original long C run diverged on later questions; a fresh short C diagnostic matched Python on its 40-question subset. The cause and any dependence on shared-GPU resource history remain unresolved.” Replace “switched state” with the observed task/order boundary, without naming an unmeasured mechanism. Confidence: high.

2. **REDLINE — the budget description incorrectly turns censored/loop-stopped failures into evidence that the model almost never makes mathematical errors.** The table labels all budget cases “ran out of the 8,192-token room,” and the text says “almost never gets the maths wrong.” Of A's 75 length failures, **16 stop below 8192**, and server logs record **17 reasoning-loop stops** (one coincides with 8192). B has a 3072-token reasoning-loop stop. A non-answer/loop termination does not establish that the unfinished computation would have been correct. **Replacement table label:** “budget/length termination (token cap or reasoning-loop stop).” **Replacement sentence:** “Of A's 81 failures, 75 ended as length without an accepted final answer and six stopped with a wrong answer. Length terminations include token-cap and reasoning-loop stops.” Keep these failures counted as registered. Confidence: high.

3. **NIT — diagnostic counts, indexing and tool-free wording need precision.** Diagnostic C matches B on **40/40 final texts**, as claimed, but differs from original C on **8/40 final texts**, not nine. Counting full reasoning changes as well yields ten, not nine; final text and reasoning must be named separately. The first chain is `final-4-001`: **scored question 121 / total request 123 including two warm-ups**, not request 122. The last 20 questions are not entirely tool-free: original C makes five calls on `final-5b-007` (one refusal and four OOM errors). **Smallest edit:** correct those counts and say “including many tool-free code/reading questions.” Confidence: high; direct ID-keyed comparisons.

4. **NIT — “no errors in either server log” and an apparent 300 MiB margin are too broad.** Saved C `server.log` contains tool-error entries, including four consecutive 708-character error results corresponding to the OOMs; refusals also log as errors. No engine restart or warning explaining the drift was identified, which is a narrower statement. And 32768 minus a sampled peak is not guaranteed allocator headroom: the actual exception reports ~28 MiB free. **Replacement:** “No engine restart or diagnostic identifying the drift was found; tool errors, including the four OOMs, were recorded. The sampled device maximum was 32,467 MiB; this is not a safe memory-margin guarantee.” Confidence: high.

## 2. What the divergence and diagnostic establish

The main pre-result observation is verified. Restricting to the **41 differing final replies**, first tool-call expression/presence differs on **21 questions**: 20 chains and one code-reading task. Their concatenated reasoning texts first differ at **8–1440 characters**, matching the write-up. Their request dictionaries are identical. First engine prompt token counts, zero reused tokens and first prefill chunk metadata also match for those 21. This supports **generation-path divergence before a tool answer**, rather than different arithmetic tool results causing every later difference.

Identical counts/chunk sizes are not saved token-ID or state-byte equality. Source inspection makes a tool-list/server-info explanation less likely: `McpHub.template_tools` renders only tool name, description and input schema; both backends return the same TOOL constant under configured server name `math`. The backend-specific MCP `serverInfo.name` is not included by that template-tool builder. Still, rendered input IDs/tool-schema fingerprints were not saved at each invocation, so the metadata alone should not be called a complete prompt-byte proof.

The boundary is also a change of task family and prompt/decoding demands, not only elapsed session age. In the original B/C comparison, all group-1 and group-3 final and reasoning texts match; group 2 has one final/two reasoning differences; all **40 chain reasoning texts differ**, but only 27 chain final texts differ. In group 5a final texts match 6/10 and reasoning 3/10; in group 5b both match 1/10. This is a concentrated later-block pattern, but it does not by itself identify a single permanent state switch.

The short diagnostic reproduces B on all 40 **final texts**, while changing eight original-C final texts and ten original-C reasoning texts. It contains fewer preceding requests and a different overall execution history from the original long C run. It favors a history/task/resource-sensitive hypothesis over “any computer process always changes the result,” but does not establish causality or eliminate transient nondeterminism.

Plausible candidates, and evidence needed to distinguish them if investigating further:

- **Executor allocation/cache growth and headroom:** `_emb` retains embeddings in `emb_cache` across machine resets. That is a concrete persistent allocation path; the OOM record shows ~68 MiB live Torch allocation. Per-call cache size/allocated/reserved memory, plus a matching device-memory timeline, would show whether growth precedes the divergence. Its existence is not proof that it changes Strata's arithmetic or execution path.
- **Strata K/V/checkpoint state:** compare selected checkpoint IDs, resumed token positions, live/host K/V layouts, GDN/PLE/indexer fingerprints and per-round allocation decisions at the first differing question. Inspected logs contain no explicit elastic-KV resize/shrink diagnostic explaining this event; absence of such a line is not proof that layout/state behavior stayed identical.
- **Expert-slot lending/refill:** counts can match while contents/order/readiness differ. Slot residency/content checksums and stream-completion markers would discriminate this, while retaining the observed 12,288-slot configuration.
- **MTP/suffix/verification paths or CUDA concurrency:** record draft source, accepted tokens, window shapes, graph-capture state and synchronized state at the first differing generated token. Matching headline acceptance counts or a fixed temperature does not certify identical intermediate arithmetic.
- **Prompt/tool rendering:** save actual input token IDs and tool-schema hashes. Current source/request evidence argues against backend identity being intentionally inserted in the prompt, but direct fingerprints would settle it.

An identical “660 MiB free” startup header is not evidence of equal memory available throughout decoding; C's separate process subsequently occupies GPU memory. A controlled long-run replication or one-factor instrumentation could test these hypotheses later. None is needed to correct the present write-up, and no new diagnostic/GPU run was requested or launched by this review.

## 3. The OOM event

It is fairly disclosed as one affected question with **four failed computer calls**, recovered by the model, and it remains in the raw data. Under explicit v3 rules, those soft execution errors can coexist with a correct final answer. The first primary comparison is unaffected because this is group 5, outside the 160 primary questions.

The event is strong evidence that this layout lacks a robust allocation margin. It occurs near the end, whereas the observed divergence begins much earlier, so it cannot retrospectively prove what caused the first divergent chain. It is consistent with a resource-history hypothesis and worth noting, not a causal diagnosis. A higher reserve was successful on earlier stage-4 prompts, but that changes expert residency; its stability/performance over the full stage-5 workload has not been measured here. Do not describe it as a proven remedy for the divergence.

For call accounting, **534 C MCP call attempts** comprise **487 complete reference-clean responses**, **43 refusals**, and **4 execution errors**, including warm-ups. Zero observed trace mismatches is valid; 534 is not a count of 534 successfully completed circuit calculations.

## 4. Does the equivalence verdict survive path variation?

**Yes, as the registered end-to-end benchmark result.** The original paired table is unchanged: 153 both right, 3 C-only, 2 B-only, 2 both wrong. The interval [-2.98, +4.41] points lies strictly within ±5. All five discordant correctness cases are chains. The evidence supports practical accuracy equivalence under the chosen margin and the specific pipeline, with observed C request latency ~1.63x B.

Path variation limits attribution: the one-question C advantage cannot be credited to superior calculator semantics, and the result is not proof of perfect model/pipeline determinism. The interval accounts for the declared question-level paired outcomes under its assumptions; it does not estimate unmeasured between-launch/session noise from one main run per arm. Retain that limitation and the post-hoc diagnostic. Do not replace original C's `final-4-029` with its successful repeat or claim a new primary interval for a hand-picked combination of attempts.

## 5. Recovered counts and failure policy

B 37 / C 43 recovered questions are reproduced. They do not hide a registered hard failure in these data: hard checks remain applied before final-answer correctness, and no recorded trace mismatch/harness/transport/fence stop was found. The four OOMs are execution errors returned through successful HTTP responses, which v3 treats as recoverable soft backend faults; explicit HTTP OOM errors would instead be hard failures. This distinction should stay visible.

Refusals and firmware `error (too complex)` responses are not successes of the tool; a later correct model answer is a success of the whole question under the frozen rule. The raw call logs and recovered column preserve that difference. Group-2 primary inclusion and the 150-question sensitivity breakdown remain as predeclared. No post-hoc exclusion of difficult cases is justified.

## 6. Does the finish-line conclusion hold?

The finish line is answered fairly **for this server-side MCP implementation and frozen synthetic benchmark**: Flash-Next calls the checked computer, both tools improve the measured task outcome substantially over alone, and C shows no demonstrated practical accuracy advantage over Python while adding latency/memory and the observed OOM episode. The property of reference checking is distinct from a measured accuracy gain. Stored-state advantages were not tested because every call starts a new calculator session, as the write-up acknowledges.

Prefer “No accuracy advantage over host Python was established within the registered ±5-point tolerance” to an unqualified assertion that C can never be better. The positive upper confidence limit still allows small differences, and the experiment is not every possible firmware/task design. Equally, a clean short repeat does not erase the resource fault or prove general state isolation.

**Final review verdict:** all reported primary scores and the pre-registered equivalence/practicality calculation check out. Fix the causal/determinism and budget-censoring prose (findings 1–2), plus the smaller repeat/index/log wording errors. The numerical result does not need a replacement run to support its bounded conclusion. Review only; no fresh model/GPU work, public posting or product-branch merge was performed.
