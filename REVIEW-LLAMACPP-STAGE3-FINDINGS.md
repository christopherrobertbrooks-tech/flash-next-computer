# llama.cpp stage 3 review

## Pass 1 — blind

Reviewed `08079dd` on `review/codex-llamacpp-stage3`. The results narrative,
second-pass questions and commit messages after a335797 were not opened before
this pass was committed and pushed. Review only: CPU checks and read-only
remote access, no model loads, installs, deployment edits or GPU jobs.

### Findings

No blocker or redline against the saved run's accuracy/latency validity.

1. **NIT — disclose the initial memory-sampling gap and distinguish load time
   from total startup.** The registration describes periodic memory sampling
   through the run; startup reports zero unavailable queries and a 0.5-second
   median interval. However, the first two samples are **132.164 seconds apart**:
   17:26:52.832 to 17:29:04.996 (Chicago). In run_split.py the initial sample
   precedes the startup identity/hash work, and the sampler thread starts only
   after that work. There are no periodic samples in between. `load_seconds`
   is calculated before that hash work too, so its 30.6 seconds is not the
   total interval from the stage-3 launch to the first question.

   This gap is **before generation**, not an unobserved question: the first row
   finishes at 17:29:12 with 7.13 seconds of timed work. Subsequent sample gaps
   are at most 0.505 seconds, apart from a short final interval; all question
   generation and helper shutdown have periodic coverage. No failed query or
   primary-result invalidation follows from this startup gap.

   Smallest reporting fix: “The maximum observed combined footprint was 17254
   MiB. There were no failed queries, but a 132.164-second startup gap separates
   the initial after-load sample from periodic sampling; generation/shutdown
   were sampled at about 0.5 seconds.” Call 30.6 seconds model load/warm-up,
   not total startup. For future runs, start the sampler immediately after
   its initial sample and report total startup separately.

### Registration and identity audit

- All **106 runtime files** currently match the frozen manifest on the gateway,
  including the bf16-head GGUF, helper, dynamic libraries, executor sources,
  runner/client, firmware, prompt assets and numerical-gate inputs.
- All **nine analysis files** match stage3-analysis.json, and their bytes also
  match its pinned source commit 3ff0dbef17f11fe6b63f089579c21a2eadcbd342.
  The analysis verifier succeeds. The relevant commit timestamps (metadata
  only) are 16:01:59/16:02:00 Chicago, before the logged 17:19 session start.
  This resolves the prior analysis-identity omission. Current HEAD need not
  equal the frozen source commit because data/review commits came later.
- Independently reproduced the registered scorer's complete output object,
  including rows and diagnostics, through stage3_analysis.py's verify-then-score
  path. No production source change was made for rescoring.
- The gateway's raw replies, startup, memory samples, preflight replies/gate
  and question hash match the committed uncompressed bytes exactly.
- One logged V100 lock session runs identity verification, the exact-helper/
  bf16-head preflight, and the 160-question run in that order, then restores
  clocks and releases. It records clock 1380 MHz, 106 matching artifacts and
  PREMISE PASS. CPU re-evaluation of the actual saved preflight replies against
  the frozen fixtures/reference files passes the gate again.
- There are **160 unique S IDs, in registered order**, matching the C order and
  registered question checksum e3f98862352ad0111c5b5428a3580d01ebc0fcec3474b8a495300e577ca7074d.
  All prompt-ID hashes match C. The source and startup preserve 8192 text,
  2000000 chip, separate 2000000 reference records per call, context 16384,
  greedy tie/EOS rules and the executor-only circuit.
- Startup status is ok with no reason/error. The log has exactly 160 progress
  rows and one successful completion. The visible stage3 inventory contains
  one stage3-full folder, not an alternate attempt. The data/logs show no
  restart, substitution, per-question rerun or selective omission. This is an
  evidence-based consistency check, not a claim to independently prove the
  absence of every unrecorded action.

### Independent answer and statistical reconstruction

Parsed the last ANSWER line independently from each saved final response and
compared its exact Fraction value against the registered answer, applying
transport/harness, incomplete response, budget/open-call and trace-mismatch
hard failures. **All 320 correctness decisions agree with the scorer.**

| Quantity | S | Saved C |
|---|---:|---:|
| Correct | 131/160 | 135/160 |
| Group 1 | 69/70 | 69/70 |
| Group 2 | 10/10 | 10/10 |
| Group 3 | 34/40 | 32/40 |
| Group 4 | 18/40 | 24/40 |
| Median outer seconds | 10.005 | 16.770 |
| Sum outer seconds | 5147.63 | 7620.77 |
| Processed text tokens | 200331 | 182707 |
| Processed/checked chip tokens | 1552141 | 1615594 |
| Text bucket seconds | 4177.32 | 6438.53 |
| Chip bucket seconds | 897.06 | 943.65 |
| Text tokens / text bucket s | 47.9568 | 28.3771 |
| Chip tokens / chip bucket s | 1730.2533 | 1712.0691 |
| EOS / text-budget stops | 156 / 4 | 156 / 4 |
| Harness / lock errors | 0 / 0 | 0 / 0 |

Paired table: **both right 121, S only 10, C only 14, both wrong 15**.
S minus C = **-0.025**, or -2.5 percentage points. A separate implementation
of paired Newcombe method 10, including the positive-covariance correction,
gives 95% CI **[-0.08765779045, +0.03707990193]**, or **[-8.7658, +3.7080]
percentage points**. It crosses zero and is not contained inside ±5 points:
**inconclusive**, neither a demonstrated difference nor equivalence. It also
does not establish a maximum 5-point accuracy loss.

The registered S-relative latency rule passes: 10.005 <= 2×16.77 and <=60.
Observed median speed ratio C/S is **1.67616x**; summed-time ratio is **1.48044x**.
This rule is a permissive practical ceiling, not an independent speed or
accuracy significance test. Group-wise gains/losses remain descriptive.

### Stop/error/timing audit

S's four text-budget failures are final-1-060 and final-4-007/019/027. C's are
final-1-060, final-3-014 and final-4-007/015. No computer-budget stop, helper
failure, lock mismatch or missing row is hidden behind a successful verdict.
S's largest prompt is 726; maximum processed text is 8192 and chip count 56409,
well inside the context/registered chip limits. There are no open tool calls.

The scorer labels two S failures “backend”: final-4-004 and final-4-033 both
contain `error (too long: split it)` tool responses with valid Python reference
values. These are recorded tool/interface limitations followed by wrong final
answers, not evidence of a CUDA/helper crash or a chip trace mismatch. Thus
“zero lock errors” is supported; “no tool errors” would be too broad.

The longest S questions are its budget stops (roughly 172–182 seconds) and a
long chain with 5733 text / 56409 chip tokens (153.31 seconds). These lengths
explain the large latencies. Adjacent row completion times reconcile with
question timings within one-second timestamp precision (residuals -0.84 to
+0.98 s); no extra multi-second inter-question pause appears. The helper log
has no error/failed/non-finite/OOM entries. Its legacy internal eval counters
are zero/infinite and cannot be used as an independent decode timing probe.

### Memory reconstruction

All **10302** samples have valid sums, zero query failures, and consistent
per-process arithmetic: **10298 run / 4 shutdown**. Maximum simultaneous sum
is **17254 MiB = 16.8496 GiB**, from parent 1714 plus helper 15540 MiB. After-load
is 17084 MiB; parent-before-helper is 1612 MiB. The shutdown maximum is 2066 MiB,
and the final sample has parent 1714 / helper absent. Per-process maxima happen
to coincide with the combined maximum here; the method correctly uses each
sample's sum rather than assuming that generally.

The UUID is the registered V100 UUID. Median actual interval is 0.5 seconds;
the startup exception is finding 1. This is a sampled process footprint,
not a true peak, and **there is no like-for-like C memory comparison**. Parent
PyTorch peak allocated 1.28 GiB cannot substitute for total process memory.

### Interpretation

The registered full run completed with reference-checked circuit traces and
an observed latency advantage under the combined engine/precision/process-split
change. Accuracy remains inconclusive. S generates about 9.6% more text tokens
and fewer chip tokens than C, so trajectories/work differ: the outer median,
summed times and bucket throughput answer distinct questions and do not isolate
an engine-only causal speedup. This does not demonstrate native circuit
execution inside llama.cpp, generalize to a different firmware/checkpoint, or
establish timing repeatability from one historical pair of runs.

The saved result is valid for those claims. No additional GPU rerun was needed
for this review; none was started. The preflight/clock-release evidence above
comes from the recorded run, not newly launched inference.

## Pass 2 — guided

Opened `REVIEW-LLAMACPP-STAGE3-PASS2.md` and the results narrative only after
blind commit `af2b8ce` was successfully pushed. The page's primary scores,
paired counts, interval, medians, totals, stop counts, chip/text rates and memory
maximum all reproduce. Independently confirmed 81/160 identical full generated
texts as well. No new blocker or redline.

### 1. Wording and scope

Finding 1 applies to the page's memory bullet: “every 0.5 s through shutdown;
10302 samples, 0 unavailable” needs the startup-gap disclosure. The sample
maximum is correct; zero failed queries does not mean zero sampling gaps.

2. **NIT — keep the tie explanation within its tested scope.** The page says
   “The free runs differ where the split takes the other side of a tie (stage
   2's checks),” then says each divergence's cause isn't isolated. The latter
   qualification is right, but the first clause can still imply that all 79
   different full-run texts have an established tie explanation. Stage 2 checked
   nine divergent development fixtures, not every full-run first divergence.
   Smallest replacement: “Full generated text is identical on 81/160 questions.
   Stage 2's nine divergent fixture runs first differed at an HF tie; this
   full-run comparison does not isolate the cause of each divergence.”

“The computer stayed exact” is fair shorthand only with its accompanying scope:
zero lock errors over the **observed** 1552141 checked chip tokens. Prefer the
literal statement “All 1552141 emitted chip trace tokens matched the RV32
reference.” This is not a universal proof over every possible input and does
not mean the final answers or all tool requests were correct.

The 1.68x median, 1.48x summed and 48.0 vs 28.4 text-bucket lines are supported.
Retain their current labels and one-run caveat; do not turn them into “every
question is 1.7x faster” or an engine-only effect. Chip bucket throughput is
similar (1730 vs 1712), rather than numerically identical. Group 4's 18/40 versus
24/40 is worth reporting as descriptive; it is not a registered subgroup test
or evidence of an identified failure mechanism. For accuracy, explicitly say
“a loss greater than five points remains compatible with the interval,” rather
than mentioning only a possible small loss.

### 2. Proposed public paragraph — draft only

> I tested a split Qwenputer runner on a V100: Qwen's layers run in llama.cpp
> (f16 with a bf16 output head), while the computer uses the existing fp32
> executor. Across 160 questions, all 1,552,141 emitted chip trace tokens matched
> the RV32 reference. Against the saved HF-bf16 executor baseline, median
> question latency fell from 16.77 to 10.005 seconds (1.68x), and summed time
> from 7,621 to 5,148 seconds (1.48x). Accuracy was 131/160 versus 135/160; the
> paired 95% interval for the difference was -8.8 to +3.7 percentage points,
> establishing neither an accuracy difference nor equivalence. This is one run
> per configuration, combining engine, precision and process-split changes;
> the circuit itself has not been ported natively into llama.cpp.

This is an appropriate qualified version of “the port works.” It is a draft
for Chris's word-for-word approval, not a message that has been posted or sent.
No upstream issue or comment was created.

### 3. Stage-4 write-up

Include the frozen protocol/source and both identity manifests, exact hardware
and locked clocks (both C and S logs specify 1380 MHz), commands and raw data,
question/order checksum and prompt-ID matches, preflight results, complete
paired table/interval, stop/failure layers, and timing boundaries. Distinguish
model load/warm-up from the additional identity-verification/startup time and
preserve the memory-gap caveat.

State that the bf16 head was selected on development data before the full run;
its unchanged stored values and fixture argmax agreement do not prove equal
free-running outputs. Original C used BF16 Qwen layers, while S uses the widened
Qwen half converted to F16 plus the BF16 head. The circuit remains upstream's
fp32 executor, checked against the RV32 reference; its checkpoint bf16 circuit
layers are not being executed in this benchmark. The disclosed upstream loader
patch loads directly to the GPU to fit host RAM; its source identity is frozen,
and SplitChat does not use that full-HF loader.

Keep C's historical PyTorch allocation peak out of any process-memory savings
claim. Report S's sampled footprint on its own. Explain that the historical
comparison and differing trajectories leave repeatability and causal
engine-versus-precision attribution unresolved. A successful execution port
and inconclusive answer accuracy are compatible outcomes.

### 4. Further measurements

**No further measurement is necessary for this registered result and its
qualified write-up.** Do not selectively rerun difficult or disagreeing answers
to alter the primary data. A repeat of a few questions could support a new,
limited timing-variability claim, but cannot repair or establish accuracy
equivalence on the registered set.

If a separate timing follow-up is wanted, choose its questions and repetitions
before running it, repeat both configurations under the same conditions in a
counterbalanced order, and record output/token checksums, startup treatment and
variability. A fixed-history teacher-forced comparison would answer a different
question about matched-work throughput. Neither follow-up is required to publish
the present observed ratios with the one-run caveat, and neither was launched
in this review.

### Disposition

The registered numerical result is reproducible and the saved run is valid for
the scoped claims above. Address the two wording/reporting nits in the final
write-up; no source fix, data substitution or additional GPU run is required.
Both passes were CPU/read-only review work. Nothing was posted externally or
merged into the source/main branch.
