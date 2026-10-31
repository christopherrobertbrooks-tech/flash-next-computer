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
