# llama.cpp stage 1 review

## Pass 1 — blind

Reviewed source `2fa7d34473dba6ef177f0854ed05bd043a0b1141` on
`review/codex-llamacpp-stage1`. Neither the narrative nor the guided questions
was opened before this pass was committed and pushed. Review only; no production
code edits, installs, or GPU/model loads. Existing gateway dependencies were used
for CPU controls, with CUDA hidden and scratch confined to an owned temporary directory.

### Findings

1. **REDLINE — numerical failures can escape the fail-closed contract.**
   Brief: “does the run fail closed on helper errors?” Client docstring:
   “a bad reply raises HelperError.” In contrast,
   `qwen_half_helper.cpp:127-136` compares logits without checking finiteness,
   and `helper_client.py::_reply` returns the five IDs/values without validating
   them. NaNs are silently skipped by the helper; entirely invalid logits can
   return `-1` IDs and infinities with success status. A length-correct reply
   containing EOS and five NaNs is accepted by `_reply` (CPU reproduction using
   a BytesIO protocol frame); `LlamaBackend` then trusts its first ID and ignores
   the values. This is a missing guard, not evidence of corruption in these runs.
   Smallest fix: fail on non-finite model output before choosing argmax; reject
   invalid IDs/non-finite reply values in the client, and add negative controls.
   Preserve the existing failed-question row and nonzero exit behavior.

2. **NIT — position controls exercise the fake backend, not LlamaBackend.**
   Brief: “Do the controls show it, and can they fail?” The test's position
   assertion reads `FakeBackend.fed`; the production `LlamaBackend` is bypassed.
   The integration fake helper consumes positions without asserting their values.
   Thus a production position/reset regression could leave these controls green.
   Current production code does reset and feeds consecutive positions correctly.
   Smallest fix: wrap LlamaBackend around a recording helper and assert RESET,
   prompt positions, subsequent positions, and forced-token payloads; deliberately
   offset a production position in a temporary negative control and require failure.

### Independent reconstruction

All 160 L and R records have matching ordered IDs and prompt-token hashes.
The separate answer extraction/Fraction comparison agrees with the scorer on all
320 correctness flags. Re-running the current scorer reproduces the checked-in
`scores.json.gz` object exactly, including diagnostics.

| Quantity | L | R |
|---|---:|---:|
| Correct | 129/160 | 130/160 |
| Group 1 | 69/70 | 69/70 |
| Group 2 | 10/10 | 10/10 |
| Group 3 | 33/40 | 31/40 |
| Group 4 | 17/40 | 20/40 |
| Median seconds | 4.665 | 8.630 |
| Sum question seconds | 2839.78 | 5491.21 |
| Text tokens | 206509 | 202043 |
| Text tokens / sum seconds | 72.7201 | 36.7939 |
| Trace instructions | 1554588 | 1615658 |
| Text-budget stops | 6 | 7 |

Pairs: both right 118; L only 11; R only 12; both wrong 19.
Difference L minus R = -0.625 percentage points. Independently evaluating the
paired Newcombe interval with the positive-covariance correction gives
[-6.7355593, +5.4733970] percentage points. It crosses zero and extends beyond
both ±5-point equivalence limits: **inconclusive**, not demonstrated equivalence.
The scorer's `practical: true` describes its observed point-estimate test; it
must not be read as a confidence-based equivalence finding.

The median ratio is 1.850x; the ratio of summed times is 1.934x. These describe
this historical comparison, with different generated trajectories. The token
rates above include prompt processing, forced tool-response feeds, reset/pipe
and emulator overhead in the denominator; they are not isolated decode rates.
The scorer's legacy `generated_tokens_total: 0` is not a count of these runs'
text tokens: use `run.text_tokens` as above.

### Implementation and controls

The prompt construction, EOS treatment, tool-call capture, forced emulator
response including HALT newline, and text/trace accounting match the current
inside4b runner. L resets per question, prefills at 0..P-1, then feeds each
processed generated token, including forced response tokens, at successive
positions. EOS is emitted but not fed/count-charged. Token-mode positions use
llama.cpp's token-position expansion; row-mode's explicit four planes are not
used by this run. Ascending token scan with strict greater-than comparisons
selects the lowest ID on finite equal logits. Reset clears recurrent/KV state.

The historical R predates the `>=` trace-cap correction in current run_plain;
this is not literally an unchanged historical runner. It cannot affect this
comparison: maximum observed traces are 52502 (L), 47127 (R), far below 2000000.
Call diagnostic fields also gained expression/instruction bookkeeping; these
changes should not be described as extra model-call differences.

CPU controls passed all six cases, including the actual two-call case (two
calls observed), text cap, exact trace cap and one above, overlong capture,
forced HALT newline, child death, explicit error, truncated reply, cleanup of an
END-ignoring child, and successful/failed runner integration. Mid-run helper
death produced exit 1 with a harness-error row and no later questions. Equality
controls cover the current HF loop, not the historical implementation. The
remaining coverage limit is finding 2. No finite-output/error evidence in the
raw records contradicts the recorded scores; finding 1 concerns unguarded cases.

Scorer changes only extend arm handling and generalize comparison naming;
answer classification remains unchanged. The reconstructed pairs, interval and
full output match. Generic “computer better/worse” verdict labels would be
awkward for L-vs-R, though the actual verdict here is inconclusive.

Timing starts before reset/prefill and ends after generation, excluding model
load, decode/write, and subsequent memory sampling. L's memory field is a
maximum of after-question nvidia-smi samples (11074 MiB), not a continuous peak
and not comparable directly with R's PyTorch peak allocated statistic.

### Interpretation

Stage 1 supports a faster plain Qwen3.5-4B baseline with an external RV32
emulator, under the combined HF-bf16 to llama.cpp-f16 change specified by the
plan. It does not isolate precision from engine, establish equal accuracy,
measure the widened circuit model's speed, or demonstrate the split circuit
port. No repeated timing trial or causal attribution follows from these single
historical runs. The numerical guard should be fixed before relying on the
same helper as a fail-closed split-runner component.

## Pass 2 — guided

Opened `REVIEW-LLAMACPP-STAGE1-PASS2.md` and the results narrative only after
blind commit `50b81e2` was successfully pushed to origin.

### Narrative and additional findings

3. **NIT — qualify the speed conclusion and divergence explanation.**
   The page says “Speed is the robust result” and “f16 and bf16 pick differently
   at near-ties.” The measured speed advantage is substantial and supported by
   both aggregate token throughput and median question latency. However, this
   is one run per configuration, with different trajectories, and does not
   establish timing repeatability or attribute each free-running divergence to
   precision/near-ties. Replace with: “The observed speed advantage is substantial
   in this comparison: aggregate text throughput is 72.7 versus 36.8 tokens/s and
   median latency is 4.67 versus 8.63 seconds. Engine and precision changed
   together; trajectories differ, and timing repeatability was not measured.”
   Replace the causal divergence sentence with: “Generated texts differ on
   79/160 questions; these runs do not isolate the cause of each divergence.”
   Independently verified 81/160 identical raw texts. Accuracy wording is already
   correct: neither difference nor equivalence is established. No correction to
   the page's numerical table is needed.

### Answers to the guided questions

1. The page agrees with the independent totals, paired outcomes, confidence
   interval and failure counts. The combined-change caveat and memory caveat
   are appropriate. State that tokens/s includes the full question timing
   boundary, rather than implying a model-only decode benchmark. The remaining
   wording qualification is finding 3. The CPU controls pass, but do not close
   findings 1 and 2.

2. Using current main's emulator/generation copy is acceptable for this saved
   comparison, with the disclosed changes. Neither L nor R reaches the trace
   cap, so the historical `>` versus current `>=` difference cannot determine
   an answer here. Preserve the exact-cap controls for future runs and record
   the deployed runner/emulator hashes alongside model/helper hashes so the
   compatibility argument remains reproducible. Do not retroactively claim
   every line is identical to the historical R runner.

3. For stage 3, sample device memory with the same nvidia-smi query for both
   configurations. Identify each cooperating process by PID and pinned GPU
   UUID; sum their memory at each sample, then take the maximum of the sums.
   Do not sum independent per-process maxima from different instants. Record
   baseline, after-load, and sampled maximum, the sampling cadence and scope,
   and cover prefill, generation and shutdown. Label it a sampled process
   footprint, not an exact allocation peak; retain per-process values too.
   Missing/failed queries must be recorded as unavailable, not converted to
   zero (the present runner uses `None`/`or 0` in its samples). Re-measure the
   original comparison configuration with that same method if a like-for-like
   claim is wanted; the historical 8.98 GiB number cannot be converted into it.

4. Before stage 2, fix the finite-output/reply guard (finding 1) and strengthen
   the production-backend checks (finding 2). Retain the plan's A,B,A reset,
   row-bit-equality, prefill/step-history, circuit ownership, chip-position and
   forced-echo/HALT controls. These must exercise the split path, not infer its
   correctness from stage 1's token mode. Verify that helper exceptions, premature
   termination, lock mismatch and budget exhaustion cannot become successful
   scored answers. Check subprocess exit status on normal context exit too:
   `Helper.close()` returns it but `__exit__` ignores it. A bounded request
   timeout would also improve handling of a live child that never replies;
   currently read calls can block before the close timeout is reached.

   Scheduler diagnostics are worth adding as a separate, short diagnostic run.
   The pinned local `ggml-backend.cpp` does read `GGML_SCHED_DEBUG` as an integer;
   use `GGML_SCHED_DEBUG=2` to inspect placement, including prompt and one-token
   decode paths. Save logs and account for deliberate host operations and
   transfers. Do not include diagnostic overhead in timed results or treat
   mere CUDA visibility as proof every expensive operation runs there.

   A bf16 head is a worthwhile optional memory experiment, not a stage-2
   correctness shortcut. Read-only safetensors metadata confirms the source
   tied embedding is BF16, shape [478656,3968]. The cut copies its first 248320
   rows unchanged. Representing those source values in bf16 rather than f32
   preserves their stored values and halves that tensor's storage (about
   3.67 to 1.84 GiB); verify the converted tensor actually round-trips bitwise.
   This does not guarantee identical matmul arithmetic, argmax, device
   placement, actual resident memory, or V100 speed. Check the pinned backend's
   supported path, inspect placement, and rerun the same fixture/argmax gates
   before adopting it. Record a distinct GGUF hash and test it as a separate
   variant; do not silently change the baseline or all tensors at once.

### Disposition

The saved stage-1 numerical result is reproducible. Its interpretation is a
promising speed observation with inconclusive accuracy, not a completed
Qwenputer port. One fail-closed guard needs correction before stage 2; the
production-backend control and wording improvements are specified above.
No GPU jobs were launched, so this review acquired no V100 lock or model memory
and had none to release. All gateway execution was CPU-only; no source/deployment
files were edited there. Only this findings document is changed in the review
branch.
