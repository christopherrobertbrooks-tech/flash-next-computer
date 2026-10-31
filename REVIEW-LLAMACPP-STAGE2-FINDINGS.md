# llama.cpp stage 2 review

## Pass 1 — blind

Source: `fb269125c273f35cf8eb413bec4b7fa24fae9dc0`, branch
`review/codex-llamacpp-stage2`. Neither the results narrative nor the second-pass
questions was opened before this pass was committed and pushed. Review only;
no installs, deployment edits, remote writes, or GPU/model loads.

### Findings

1. **REDLINE — the free-running GPU control can pass an immediate-EOS helper,
   and leaves the divergent suffix unverified.** Brief: “Anything a split-specific
   failure could slip past … a helper that silently changes?” Plan: “prefill
   and step-by-step histories agree on the same fixture.”
   `test_split_gpu.py:89-102` checks rows only for `gen_index < d`, where d is
   the first token divergence, then accepts EOS and no lock error. Replaying
   this exact success predicate on CPU with a helper returning EOS at its first
   prediction yields True: d=0, zero step rows checked, even with an HF margin
   of 1.0. Repeating that same EOS also passes the final A-repeat predicate.
   The test prints the divergence margin but never gates on it. This does not
   show the real helper did this: the saved run has many substantive calls and
   nonzero histories. It shows that this test alone cannot certify a changed
   helper. CPU stream tests use a different scripted FakeHelper.

   A second limit is substantial in the real logs: after divergence, rows no
   longer get compared against any independent executor/HF history. For
   final-2-003 only 8 of 4259 text rows are compared; for final-4-003, 15 of 879.
   A CPU negative fixture with a corrupted post-divergence row still satisfies
   the unchanged prefix predicate. The lock verifies circuit output tokens;
   it does not inspect the entire residual row delivered to Qwen.

   Smallest complete fix: bind the numerical premise gate to the exact helper,
   library and GGUF used for stage 3, and require its registered agreement/margin
   thresholds, rather than accepting arbitrary free divergence as a helper test.
   Add a control that replays an actual divergent split history through the
   independent upstream executor/HF path and compares every `_exec_hc` row on
   that same history, including the suffix. Preserve prefix comparison against
   the old fixture; do not compare different histories as though equal rows were
   expected. Negative controls must include immediate EOS and suffix corruption.

2. **REDLINE — the memory implementation does not yet implement the registered
   V100/shutdown scope.** Plan: “each cooperating process by PID on the V100's
   UUID” and “prefill, generation, shutdown.” `run_split.py:39-41` queries only
   `pid,used_memory` across all GPUs, making a dictionary keyed by PID. A process
   present on two GPUs gets one arbitrarily last value. CPU reproduction with
   repeated PID entries gives 1700 MiB instead of the intended V100 sum of
   300 MiB. CUDA visibility is not a filter for nvidia-smi queries.
   Additionally, the sampler is stopped before leaving the Helper context, so
   helper shutdown is excluded. `join(5)` is shorter than the query's 10-second
   timeout; a slow query can still be running while its report is written.

   Smallest fix: query/filter the pinned GPU UUID (and record it), keep sampling
   through helper shutdown with an explicit policy for expected exited PIDs,
   then join the sampler completely before saving a consistent report. Record
   timestamps/actual cadence; the current 0.5 s wait is added after query time.
   The saved pilot's summed samples are useful as reported observations, not
   proof that these missing scopes were measured. No evidence that a second
   GPU actually contaminated this pilot is asserted.

3. **NIT — the pipe-cost comparison does not isolate the pipe's share.**
   `bench_helper.py` says to compare with llama-bench “to get the pipe's share.”
   It compares warmed row-input steps at positions after a 447-token prompt
   against token-input tg128 at depth 0, with differing context/batch details.
   The helper reports median 22.02 ms / mean 22.03 ms versus approximately
   21.998 ms from llama-bench; bf16-head is 19.85 / 19.87 versus 19.870 ms.
   These support comparable end-to-end step throughput, not a measured
   subtraction estimate of pipe/serialization overhead. No confidence interval
   for the difference is supplied, and a negative estimate is not negative cost.
   Smallest wording fix: “Compare end-to-end helper step throughput with an
   unmatched llama-bench reference; this does not isolate pipe overhead.” For
   an actual overhead estimate, benchmark the same row history, context and
   decode settings with/without transport, or instrument matched decode and
   round-trip timings with performance collection enabled.

### Source comparison

Compared against the gateway's `chipformer/rv/chat.py` without changing it.
For executor=True, fast=False, the copied loop preserves ownership, lock
ordering, chip/text budgets, uncounted EOS, and text-only Qwen positions. The
helper receives a row for every processed text token, including forced byte
responses and HALT newline; chip-only inputs advance the executor but not Qwen.
Circuit-owned predictions override the helper. Unexpected chip ownership and
wrong chip tokens cause a lock error; a given-up chip position raises.

The snapshot is installed before executor prompt processing. exec_prompt_rows
mirrors ExecRow.start: capture if needed; clear S/history; process prompt; then
load snapshot initial states while retaining prompt lag history. Its additional
row construction allocates a fresh bf16 row each time. SplitChat's `_exec_hc`
uses the same checkpoint embedding prefix, zero residual elsewhere and the same
kept columns; no full HF checkpoint is loaded. It inherits upstream's executor,
prompt, token decoding and text rendering methods. The original HF prefill
precedes snapshot installation, whereas split's replacement derives prompt rows
after snapshot; the recorded fixture comparisons support this replacement for
these prompts. Saved state checks compare S and lag histories with upstream
start, not a separate full-history suffix oracle.

`head_mismatch=None` is honest: the helper lacks the full widened-vocabulary
head needed for that diagnostic. `verify_mismatch=None` matches verify=False.
The split's returned seconds/chip/text buckets exclude prefill just as upstream
loop timing does; run_split's outer question seconds include reset, prompt-row
construction and prefill. Reference-machine setup/history and output parsing
are outside the outer timing. Do not compare an inner bucket with an outer
question latency as though their boundaries were equal.

### Checks and raw observations

- Hash-matched deployed `test_split_chat.py` and `split_chat.py` to the reviewed
  files. Ran all CPU controls read-only with CUDA hidden: one/two real reference
  call streams, budgets, circuit ownership, wrong chip token and given-up
  circuit position passed. An in-memory helper position offset made five
  position checks fail and exited 1. No remote temporary files were needed.
- The checked-in GPU log records prompt-row equality and S/history equality for
  12 fixtures, EOS/no lock error on their free runs, and first-A repeat. It is
  meaningful evidence within its prefix boundary, subject to finding 1. No new
  GPU control run was performed in this review.
- Pilot: 12 unique records, all EOS, no harness/lock errors; 536.01 summed
  question seconds, median 25.01 s; 18203 processed text tokens, 188945 chip
  tokens. Chip/text buckets sum to 110.45 / 418.63 s respectively. Different
  generated trajectories mean these bucket totals are not a fixed-work engine
  comparison. EOS and a correct computer do not imply the final answer is right.
  This pass does not claim an independently graded pilot accuracy.
- Pilot runner/split hashes match reviewed source. Startup records helper
  `acc89740...`, GGUF `b0f08db8...`, 8192 text / 2000000 chip budgets and a
  per-call lock cap. It reports parent-before-helper 1612 MiB, after-load sum
  18964 MiB and sampled maximum sum 19108 MiB, 1030 valid / zero unavailable
  samples. Parent PyTorch peak 1.25 GiB is a different quantity.
- Placement logs for both head variants each contain 16100 CUDA0 node entries
  and 28 CPU GET_ROWS entries across repeated diagnostic/reservation graphs.
  The large result_output matmul is on CUDA0 for prompt and one-token shapes.
  CPU embedding lookup/mapped input copies exist; this is not “every op on
  GPU.” These counts are repeated logged graphs, not runtime op counts.
- Baseline llama-bench tg128 is 45.458928 tokens/s; bf16-head is 50.327814.
  Reported model bytes drop from 15009728000 to 13039060480: a 1970667520-byte
  (1.835 GiB) reduction. The helper's warmed rates are 45.4 and 50.4 steps/s.
  Diagnostic helper perf prints have zero eval times / infinite rates, so those
  counters cannot independently validate latency or overhead.
- CPU memory-mapped comparison of the actual GGUFs confirms only
  token_embd.weight changes tensor type; all other tensor values are equal.
  Correctly interpreting GGUF's raw BF16 bytes as uint16, every bf16 head value
  expands bit-for-bit to its baseline f32 value. Stored-value preservation is
  established; identical runtime arithmetic/answers does not follow. The saved
  teacher-forced bf16-head premise summary reports 99.782% split vs 99.796%
  control on 13746 Qwen-chosen positions, zero high-margin disagreements and
  PREMISE PASS. This is a numerical premise, not a 160-question accuracy result.
- The viewer switch calls the split loop under a Helper context, propagates
  token callbacks and labels the executor path without claiming bf16 circuit
  layers run. It does not add a lock Session itself; the viewer's existing
  reference checks remain distinct from run_split's generation-time lock.

### Stage-3 disposition

The runner is a credible executor/llama.cpp split and the pilot demonstrates
end-to-end operation. The evidence does not establish parity over unverified
histories, a 160-question result, an isolated engine effect, or a native circuit
implementation in llama.cpp. Close finding 1 and fix the registered measurement
scope in finding 2 before treating this as a validated stage-3 run. Keep the
baseline f32 head and optional bf16 head separate; bind registration to exact
artifacts/upstream revision and the chosen variant. The stated ±5-point paired
comparison against original C and zero-lock-error requirement remain the right
questions; do not relax them based on a 12-question pilot.

No GPU lock or memory was acquired by this review. The supplied pilot, benchmark
and GPU-control logs each report clock restoration and V100 release. No GPU
process was stopped, and no claim is made that shared hardware is currently idle.
