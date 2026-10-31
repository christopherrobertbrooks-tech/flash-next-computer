# Inside-4B v2 results review

Source branch `inside-4b` at `111eb69`; frozen run code tag `inside4b-freeze-2` resolves to `125a82f9e8ef831def2ce0e73dc91e302af2d0e5`. Review only: no inference, GPU/model loads, gateway writes or use of stopped-v1 data.

## Pass 1 (blind)

**No numerical findings in the saved result.** All 320 question/arm decisions and the complete published score JSON reproduce. The registered accuracy result is **inconclusive**, not equivalence or superiority. The registered median-latency rule **passes**, narrowly on the ratio criterion. This compares the two disclosed execution systems; it does not isolate CPU placement or test checkpoint bf16 computer layers.

### Independent rescore

Parsed the last ANSWER line independently, compared exact Fractions against the frozen gold, and applied hard stop/transport/harness/open-call/lock checks. Multiple identical ANSWER lines occur in some reasoning/final replies; the registered scorer uses the last one, so requiring exactly one would wrongly reject those answers. After applying the registered policy, every independent decision agrees with `stage5/score.py`. Independently reconstructed the failure layers from calls and stop records; all agree too. Running the branch scorer with `--compare C,R` produces an object structurally identical to committed `results/inside4b-full/scores.json.gz`.

| Group | R | C |
|---|---:|---:|
| 1 | 69/70 | 69/70 |
| 2 | 10/10 | 10/10 |
| 3 | 31/40 | 32/40 |
| 4 | 20/40 | 24/40 |
| Primary total | **130/160 (81.25%)** | **135/160 (84.375%)** |

Paired table: **126 both right, 9 C-only, 4 R-only, 21 both wrong**. Delta C−R = **+3.125 percentage points**. Independently evaluated Wilson endpoints and Newcombe method 10 with the positive-covariance correction: **95% CI [−1.5496526430, +7.9668399307] percentage points**. This crosses zero and extends above +5: neither superiority nor registered ±5-point equivalence is established. Descriptive groups 1/3/4 table: 116 both right, 9 C-only, 4 R-only, 21 both wrong (150 questions); do not replace the primary denominator with it.

Median wall seconds: **R 8.63, C 16.77**; ratio **1.943221321**. C is below both 2×R and 60 seconds, so `practical=true` under the registered rule. This is a pass for these observed medians, not a confidence interval or a guarantee over repeated runs. Sum of recorded question times: R **5,491.21 s**, C **7,620.77 s** (ratio ~1.388); these are different quantities from the ratio of medians and from complete process elapsed time.

| Failure layer | R | C |
|---|---:|---:|
| budget | 7 | 4 |
| call format | 5 | 2 |
| backend | 0 | 4 |
| final answer | 16 | 11 |
| reasoning | 2 | 4 |
| Total wrong | 30 | 25 |

Recovered questions: **16 per arm**; firmware error-result calls: **R 27, C 28**; neither arm records tool execution exceptions. “Backend” here includes firmware limits where Python can evaluate the expression; it is not proof of a wrong circuit instruction or lock mismatch. Correct answers after soft tool failures remain correct under the frozen policy; disclose recovered faults separately rather than silently treating them as clean calls. Logged tool calls: R **419**, C **423**. Chain steps all seen in order: R **8/40**, C **10/40**; chain batching and local reasoning can yield correct final answers without emitting every intermediate as a separate tool result.

### Frozen protocol and provenance

- Freeze commit: 2026-10-30 20:06:38 −0500; annotated tag created **20:12:00**, master run begins **20:12:03**. Master log shows **R exit 0 at 21:43:50, C exit 0 at 23:51:13**, in the prescribed R-then-C order. Launch script runs each arm once through the V100 lock, with greedy/thinking-off settings inherited from the frozen code, text limit 8192 and computer limit 2,000,000. Saved run.out files each contain exactly the ordered 160 question IDs and a normal completion/lock release. No retries, duplicate IDs, extra rows, restart/error traces or replacement rows appear in these artifacts. This supports one recorded run per arm; it cannot prove the absence of an unrecorded external run.
- Both runs have exactly the 160 unique frozen primary IDs in order. Question SHA256 is **`e3f98862352ad0111c5b5428a3580d01ebc0fcec3474b8a495300e577ca7074d`**, identical in local file, run fingerprints and gateway deployment. Corresponding prompt-ID hashes match on **160/160** pairs, as do prompt token counts. Prior CPU tokenizer verification covered these exact prompts; no new checkpoint/model execution is needed to re-establish it here.
- Current deployed runner/frame SHA256 values match the frozen tag: R **`e0181a31c09c3b497e6e2292c1e91a4b0f3f3ff2641125a038fd31d97a18ddf1`**; C **`4a41bd1a9374c1cf57ccf8ca2c016770934fa530e75765cf212dce0f2eec8388`**; frames **`ae26ff6ef7ae708069519d313a9683d332c209f7b31b1f549d6367e7b507ca5e`**. The gateway retains the frozen `>` trace-cap test and old R logging, not the branch's post-freeze fix. This checks the present deployment against the freeze; code hashes were not saved immutably in each startup file at launch.
- Decompressed local reply bytes equal the gateway raw files: R **`63993025451f4031602aceb7e4475741f1626306e42f27c98fcec55d8fe295f1`**, C **`52106bd846493918b397c8b7ff44689bc6a216eff194ff447bc0cd02d15f0488`**. Both startup.json hashes also match local versus gateway. Each saved row's final content/open-call count recomputes exactly from its full_text with the frozen frame parser. Saved run.out ID sequences exactly match the raw rows.
- Startup settings: both max_new **8192**, both computer budgets **2,000,000**, C lock allowance **2,000,000 records/call**, reduced-precision bf16 reductions **false** in both, identical firmware SHA256 **`b643ad61291a947fa6c7498de9170b1d4da062dc1f34810feba9c4eebdec862d`**. Tokenizer hashes match the independently checked base/extended checkpoint tokenizers. C startup explicitly names the fp32 executor and says checkpoint bf16 computer layers are not executed. `verify_mismatch=null` is therefore not an HF-module verification pass.
- The only scorer diff after the tag is the R `call_text` fallback in loop/error-expression diagnostics. It changes neither answer grading nor CI/decision policy. It also prevents missing-expression access on error-result rows. Post-freeze runner changes are disclosed and were not used to claim a corrected run. Stopped-v1 was not opened or used.

### Every stop and trace-budget audit

The committed companion file **`REVIEW-INSIDE4B-TRACE-AUDIT.csv`** lists **all 320** arm/question rows with stop reason, trace tokens, text tokens and open-call count. Every row's summed per-call trace tokens exactly matches its answer-level trace counter.

| Arm | Question | Stop | Text tokens | Trace tokens | Open calls |
|---|---|---|---:|---:|---:|
| R | final-1-060 | text budget | 8192 | 36148 | 0 |
| R | final-3-002 | text budget | 8192 | 5686 | 0 |
| R | final-4-007 | text budget | 8192 | 0 | 0 |
| R | final-4-011 | text budget | 8192 | 30946 | 0 |
| R | final-4-015 | text budget | 8192 | 0 | 1 |
| R | final-4-026 | text budget | 8192 | 40881 | 0 |
| R | final-4-030 | text budget | 8192 | 21422 | 0 |
| C | final-1-060 | text budget | 8192 | 18074 | 0 |
| C | final-3-014 | text budget | 8192 | 3512 | 0 |
| C | final-4-007 | text budget | 8192 | 35671 | 0 |
| C | final-4-015 | text budget | 8192 | 42319 | 0 |

All other rows stop at EOS: **R 153, C 156**. No computer-budget stops, lock errors, head mismatches, HTTP errors or harness errors. Exactly **one open call**, R final-4-015; it is a hard failure and does not supply an accepted final answer. All eleven budget stops have exactly 8192 text tokens.

Maximum per-answer trace usage: **R 47,127** (final-2-007), **C 53,161** (final-4-022), respectively **2.35635%** and **2.65805%** of the computer cap. No answer equals or approaches two million; thus the disclosed `>` versus `>=` exact-cap issue is **not exercised**, and neither is C's record allowance. Totals: R **1,615,658**, C **1,615,594** trace tokens across all questions. Totals across the whole dataset must not be compared with a per-question cap. R precomputes whole reference calls before injecting response tokens whereas C streams them; counters describe those tested paths, not guaranteed identical interrupted prefixes.

### Interpretation conditions for the later write-up

1. State **inconclusive accuracy**, despite C's higher observed score. “No demonstrated advantage” is defensible; “equivalent,” “same accuracy” as an inferential conclusion, or “C is better” is not.
2. State that the **registered median practicality rule passes**, reversing the pre-run prediction of ~3× cost/failure of that rule. The point ratio is only ~2.8% below its 2× threshold; there is no repeatability estimate here. Do not replace the rule after observing this result.
3. Report memory as **PyTorch peak allocated**, not total board residency: R **8.98 GiB**, C **18.98 GiB**, ratio **2.113585746**. Startup allocations are R 7.83 GiB, C 17.45 GiB; these are not peaks. The saved measurement does not include every non-PyTorch allocation/reservation or establish board headroom.
4. Call this an **implementation comparison**: widened/rescaled Qwen plus fp32 circuit executor and a reference checker versus ordinary base Qwen plus reference emulator. It does not isolate CPU location, show that checkpoint bf16 computer modules ran, or establish that firmware can never fail. Zero checked mismatches means agreement over the emitted checked prefix on these questions, not a universal proof.
5. Keep stage-5 results and this run separate: different model/runtime/tool interface and question subset prevent attributing cross-stage differences to the computer alone. There is no full-run plain-alone arm and no independent inference repeat here.

Blind boundary: the results narrative, README status, post-freeze commit messages and second-pass questions were not opened before committing/pushing this pass.

## Pass 2 (guided)

Blind commit `7102005` was pushed before opening the guided brief, narrative, README status or post-freeze commit messages. Read those afterward. The headline scores, pairs, inconclusive verdict, practicality pass, memory measurements and cap audit agree with pass 1. The important disagreements concern interpretation, timing attribution and failure-layer comparability; they do not change the primary accuracy verdict.

### Findings against the write-up

1. **REDLINE — “identical computing” and the claimed cause of the accuracy split go beyond the evidence.**
   Plan: “this compares two implementations, not CPU placement alone.” Write-up heading: “Why they differ: the language model's path, not the computing”; bullets say “Same computing work” and “that changes greedy choices,” followed by “the computing was identical.” The totals **1,615,594 vs 1,615,658 trace tokens** are very close but neither identical nor a comparison of identical call inputs. There are **423 vs 419 calls**, different requested expressions/strategies, and only **114/160** question pairs even have equal trace totals (equal totals themselves do not prove equal traces). Zero C lock errors supports the executor agreeing with the reference on C's emitted checked stream. It does not show that C and R performed identical computations or identify the cause of their language-model divergence.

   Smallest replacement for the causal paragraph: “Both arms use the same calculator firmware. C's emitted circuit stream matched its reference checker with no recorded mismatches. Aggregate trace-token counts were similar, but the models often requested different calculations. Model width, rounded norm compensation and runtime differences are plausible contributors to divergent generation; this experiment does not isolate their causes or CPU placement.” Replace “leaning to inside, by path noise” with “C had a higher observed score; superiority and equivalence were both unestablished.” The no-tool divergences in the pilot show that tool outputs are not the only possible source; they do not isolate a unique numerical cause. On this run, wording/call divergence and differing final verdicts are associations, not a causal ablation.

2. **REDLINE — the failure table misidentifies the firmware error and compares inconsistent diagnostic oracles.**
   Write-up: “backend (the firmware's ‘too complex’ on a long chained expression, where Python gives a value).” All four C backend-failure rows (**final-4-001, 010, 019, 024**) actually contain **`error (too long: split it)`**, not `too complex`. They hit the capture window, not a demonstrated shortage of arithmetic registers. More importantly, R stores `python_reference = ferr or python_value(captured_expr)`, whereas C evaluates its regex-extracted whole parameter. Thus R's too-long capture error is reused as the supposed Python reference and gets called “call format,” while C's whole expression can evaluate successfully and gets called “backend.” The exact same error mechanism can receive different layer names solely because of metadata construction.

   A read-only CPU check of the saved R expressions using the same whole-parameter oracle as C found values for **final-4-017, 021, 024**, where R had saved `python_reference='error (too long: split it)'`. For R final-4-022 and 029 the whole-parameter oracle returns `error`, so not every too-long call would move category. Under that consistent whole-parameter diagnostic policy, **three R failures move from call format to backend** (R backend 3, call format 2); **no question's correct/incorrect verdict changes**. This is a diagnostic counterfactual, not permission to overwrite the frozen raw rows or silently redefine their grading.

   Smallest publication fix: label the table “failure layers from recorded metadata,” state the asymmetric reference construction and give the three-row sensitivity, or present raw firmware error categories without claiming the layers are directly comparable. Replace the C backend annotation with “recorded too-long capture errors on expressions the full-parameter Python oracle evaluates.” In a future runner, record the actual captured bytes and the intended parameter separately, use a single named oracle/input policy in both arms, and preserve raw firmware limits as their own category. Pass 1 faithfully reproduced the saved layer labels; this additional cross-arm oracle check shows why reproducing those labels alone did not validate their semantic comparability.

3. **REDLINE — the time table's “reference check included” and decomposition are not what the timers measure.**
   Write-up: “944 s (circuit executor, 1,712 trace tokens/s, reference check included)” and “the rest of C's extra time is the wider model writing its text more slowly.” Upstream generate performs `lock.qwen` and expected-token comparisons **before** `ts = time.time()` for the chip/text buckets. Therefore reference-check work is excluded from both `chip_seconds` and `text_seconds`. Executor work on text tokens is included in the text bucket, so that bucket is not isolated base-model time either. R's per-call ms surrounds whole emulator_response (including creating/parking a reference machine and collecting response text); it is not an equivalent chip-token timing boundary.

   Exact bucket sums: C chip **943.65 s**, C text **6,438.53 s**, recorded question wall **7,620.77 s**; residual **238.59 s** includes prefill/setup, reference work and other unbucketed overhead. This residual is not a measurement of reference cost alone. R tool-call ms sum **3.7686 s**. These are useful implementation measurements but cannot establish that the executor alone accounts for all computer overhead, or attribute the rest of the C/R wall difference solely to wider Qwen computation.

   Suggested replacement: “C recorded 943.65 s in chip-token steps and 6,438.53 s in text-token steps; its reference checking and other overhead lie outside those buckets. R recorded 3.77 s around complete emulator calls. The bucket boundaries differ; end-to-end wall time is the comparable cost measure.” Keep **1,712 trace tokens/s** labeled as trace tokens per chip-step bucket second, excluding checker overhead, rather than fully checked end-to-end circuit throughput. No new performance run is needed to correct these descriptions.

4. **NIT — distinguish full-text identity, prediction nonconfirmation and raw firmware faults.**
   The reported **88/160** identity count and group counts **67/70, 5/10, 14/40, 2/40** are correct for **full_text**, including calls/responses. Extracted final assistant content matches on **96/160** (68/70, 6/10, 15/40, 7/40). Specify full generated text rather than leaving “replies identical” ambiguous. The 13 correctness-discordant pairs do fall outside the 88 identical full texts, as expected; this adds no causal identification. “Both [predictions] wrong” overstates an inconclusive accuracy result: say “equivalence was not established; the predicted failure of the practicality rule did not occur.” Both arms have 16 recovered **questions**, not necessarily exactly 16 fault events; call them firmware errors/limits rather than “refusals” unless a distinct refusal mechanism is shown. For CI rounding, use the unrounded calculation: approximately **[−1.55, +7.97] percentage points**; the published −1.6 lower endpoint comes from rounding already-rounded scorer JSON and is slightly conservative, with no effect on the verdict.

### Guided answers

**1. Agreement and who is right.** All primary numbers and the formal registered outcomes agree. The raw error strings and upstream timer placement settle findings 2–3 in favor of the review; the causal phrasing in finding 1 has no identifying intervention behind it. README status does not introduce an inside-4b accuracy claim that conflicts with these findings. The post-freeze scorer change and run provenance are disclosed honestly.

**2. Computing identity.** Narrow this to shared firmware and reference-checked correctness of C's generated stream. Similar total token work is a descriptive coincidence of these workloads, not equality of expressions, instruction traces or results. The different call counts and per-question strategies are direct counterexamples to literal identical work. `final-4-012` indeed has zero C calls and twelve R calls, despite both being scored correct. Neither equal aggregate counts nor no mismatches imply that the two language models made the same requests.

**3. Cause of the 9-versus-4 split.** Numerically different Qwen paths are a plausible source of initial divergence; those choices then affect call content, token length, firmware capture limits and final reasoning. Those downstream failures can explain individual scored outcomes without proving why the initial choices differed. The “backend” cases are capture-length errors, not incorrect circuit execution; the cross-arm error classification is asymmetric. The experiment supports “no demonstrated benefit attributable to CPU placement,” not “the difference is proven path noise.” Deterministic greedy differences need not be random noise, and one run does not estimate their repeatability.

**4. Verdict.** **Inconclusive** is exactly the pre-registered result: delta +3.125 pp, interval crossing 0 and +5 pp. A point estimate leaning upward is descriptive; an advantage, equivalence, or an identified numerical-noise explanation is not established. Keep the observed score difference and uncertainty together.

**5. Practicality and pilot comparison.** “Practical, just” is fair if explicitly tied to the registered **median** rule on this run: 1.943× and 16.77 s satisfy both conditions. It does not mean similar memory, low latency on every prompt or proven repeatability near the threshold. The pilot was twelve primary questions with three per group; the full set has **70/10/40/40** across groups 1–4, so near-limit questions fell from 25% to 6.25% while short arithmetic rose from 25% to 43.75%. Prompt instances also changed, the missing HALT newline was fixed, and C's cap rose from 60k to 2m; these changes were not isolated experimentally. Full-run group medians R/C are **3.43/8.085**, **18.425/50.395**, **14.105/22.31**, **63.54/93.945 s**; pilot R/C medians were **3.70/9.03**, **57.89/124.35**, **12.63/16.21**, **102.05/198.69 s**. Those show substantial workload differences. Ratios of overall medians are not weighted averages of group ratios. None of the full-run answers hit even 60k trace tokens, but that alone does not prove that the cap fix caused the speed-ratio change. Present mix/newline/budget differences as reasons the pilot estimate was not a controlled forecast, not as a measured causal decomposition.

**6. Time breakdown.** No: timer boundaries differ, and the claimed included checker is actually outside C's chip timer. Findings 3 gives the exact sums and a replacement that preserves the valid observations. The registered whole-question medians remain valid regardless of this bucket-label error; C's GPU peak allocated remains ~2.11× R's.

**7. Post-freeze scorer repair.** Reproduced the frozen scorer's first exception on saved R **final-2-003**: `KeyError('expression')` in the diagnostic list. The unpatched frozen scorer cannot issue a complete result, so say “the frozen scorer crashed; a disclosed diagnostic-only patch was used to compute scores,” rather than implying it independently completed and agreed. The two-line fallback patch also repairs the loop flag and does not modify final-answer parsing, hard failures, paired statistics, margin or practicality. Independent regrading of all 320 rows and independent CI calculation support the result without relying on the assertion that a crashing scorer would have produced it. Preserve the tag, patch and raw data, as done here. Separately disclose the failure-layer metadata asymmetry from finding 2; this was not fixed by the fallback.

**8. Final answer to the question.** The bounded conclusion is useful, but its “computing is identical” and “difference comes from” statements need narrowing. Suggested replacement: “On these 160 questions, the checked Qwenputer executor scored 135/160 and the plain model with the RV32 reference scored 130/160. The registered accuracy comparison was inconclusive. C passed the median-latency rule at 1.94× R and used 2.11× peak allocated GPU memory. C's emitted circuit stream had no reference mismatches. Because model width, numerics and runtime also differ, this does not isolate a benefit or cost of CPU placement.” Checkable traces are a demonstrated property within the checked stream; “self-contained inside the weights” is an architectural claim not established by this executor-plus-external-checker run. It may be discussed as a design motivation, clearly separate from the measured conclusion.

### Final review status

The accuracy result and registered practicality result stand. Fix the causal wording, failure-table explanation and timing labels before treating the narrative as reviewed. No additional inference, GPU run, merge, publication or change to frozen data was performed or needed for this audit.
