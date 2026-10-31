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
