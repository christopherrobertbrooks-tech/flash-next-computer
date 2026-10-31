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

## Pass 2 — guided

Opened `REVIEW-LLAMACPP-STAGE2-PASS2.md` and the narrative only after blind
commit `dae624e` was successfully pushed. Additional reads below address the
new guided questions: original C's records, scorer compatibility, saved fixture
margins, and the current transport implementation.

### Additional findings before stage 3

4. **BLOCKER — arm S is silently excluded from the scoring pipeline.**
   `run_split.py` documents arm S as “graded by stage5/score.py,” but the scorer's
   `arms` whitelist (`stage5/score.py:400`) is A/B/C/R/L. Its comparison parser
   also permits only C or L first, and the group-4 chain whitelist omits S.
   CPU reproduction using the 12 genuine S/C pilot records and their matching
   question subset (owned temporary copies with the correct subset checksum):
   `--compare C,S` exits 0, scores only C, and emits no primary comparison;
   `--compare S,C` exits 1 with the first-arm assertion. This is not a missing-
   questions artifact: both temporary arms contain every question in the subset.
   A complete 160-row S run would still be excluded by the same whitelist.

   Smallest fix: add S to scoring and chain-arm support and allow S as the first
   comparator. Add real CLI regression checks for S,C, including both arms'
   row counts, paired table, signed interval, and the S-relative latency rule.
   Unknown/unsupported arm labels should be rejected explicitly rather than
   loaded and silently discarded. Freeze the corrected scorer before stage 3;
   do not rename S records to C or L merely to get a result.

5. **REDLINE — lock errors fail a question but do not fail the run.**
   Registration: “0 lock errors required (any mismatch fails this design).”
   `run_split.py:160-169` sets a nonzero run status only for a caught exception.
   An ordinary returned lock_error is recorded as trace_mismatch, then the run
   continues and eventually exits successfully with “done.” The scorer does
   correctly make a trace-mismatch question wrong, regardless of final text;
   that is weaker than rejecting the design as soon as any mismatch occurs.

   Smallest fix: after persisting the offending row, stop and set a nonzero run
   status on lock_error, with an explicit invalid-design reason; release helper
   and sampler in finally paths. Alternatively, if continuing is deliberately
   registered for diagnostics, maintain a sticky invalid-design status and
   withhold the primary acceptance verdict at the end. Add a runner-level CPU
   test: valid final-answer text plus lock_error must never yield successful
   design completion. Do not silently restart or omit the offending question.

### 1. Narrative claims

The pilot rates reproduce: S text bucket = 18203/418.63 = 43.4823 tokens/s;
original C on those same 12 IDs = 29.4542. Chip buckets are 1710.68 versus
1702.77 tokens/s. Pilot prompt hashes match C on all 12. These are bucket rates,
not outer latencies: S text tokens / total question seconds is 33.9602 tokens/s.
The page labels the table's text-step boundary correctly. They still describe
different generated trajectories, not a controlled fixed-history speedup.

Finding 3 applies directly to the page's “pipe add about 0.04 ms” claim. Replace
that claim with the matched units and the limitation: “Warmed helper row steps
average 22.03 ms (median 22.02); the separate token-input llama-bench reference
is about 22.00 ms. These unmatched measurements do not isolate pipe overhead.”
Similarly, the derived “rest … ~1 ms … computer side” is a residual across
unmatched workloads; do not present it as an independently measured breakdown.
Replace “runs at llama.cpp's own speed” with “is close to the separate
llama-bench throughput on this widened GGUF.” It is about 4% lower in the text
bucket, and whole-question performance includes much more than that bucket.

Read-only CPU tensor metadata gives 13.9789 GiB of stored tensors for the
widened baseline and 8.0603 GiB for plain Qwen. Thus the rough storage comparison
is reasonable in **GiB**, not GB. It is not a measured “bytes read per token”
quantity: there are no memory-traffic counters here. Prefer “The widened GGUF
stores about 14 GiB of tensors versus about 8 GiB for plain Qwen, plausibly
contributing to lower decode throughput.” It does not isolate bandwidth from
kernel shapes, precision or other overhead.

The placement description agrees with the logs: input-embedding GET_ROWS are
on CPU; output matmul is on CUDA0. Distinguish logged graph reservation nodes
from operations actually needed on row-only input. The memory table already
notes C's PyTorch statistic is incomparable, but “sampled the way Codex
specified” is premature given finding 2. State the actual limited scope until
it is fixed. The nine saved divergence positions have reported zero HF margins;
their existence does not validate arbitrary future high-margin divergence.

### 2. Proposed primary configuration

**Yes: registering bf16-head as the primary before any stage-3 data is sound,**
provided the plan is explicitly amended and findings 1/4/5 and the measurement
implementation are resolved. Selection from development fixtures/pilot timing
is legitimate; describe it as selection on development data, not a new untouched
validation set. There is no need to require two full 160-question primary runs.
Keep f32 as a reference/diagnostic variant; any later f32 full run should have a
separate preregistered role, not become a replacement primary after seeing S.

Independently compared the saved stage-0 f32-head and stage-2 bf16-head replies:
argmax IDs match at all **14745** fixture positions, and the repeat check is
true. The primary premise denominator is **13746 Qwen-chosen positions**, not
14745; the extra circuit-owned positions do not enlarge the accuracy gate.
The CPU GGUF comparison in pass 1 confirms unchanged stored head values and
other tensors. Stored equality and fixture argmax equality are substantial
premise evidence, not proof of free-run equivalence for all 160 questions.
Exact llama-bench rates imply about **10.71%** throughput improvement (roughly
10.6–10.7% at the page's precision); storage savings are **1.97066752 GB /
1.835 GiB**, not a measured guarantee of equal GPU-footprint savings.

Freeze the variant's full hash, currently reported as
`643e78a920c52bce7b32c14f5cfe2781e639aa2ff4bccbdffe971b1eeea7add4`,
and verify it against the actual file immediately before the gated run.

### 3. Concrete registration to freeze

Before launching, commit a protocol containing:

- The exact 160 question IDs/order and question bytes, SHA256
  `e3f98862352ad0111c5b5428a3580d01ebc0fcec3474b8a495300e577ca7074d`;
  no exclusions, post-hoc retries or substitution. Specify a separate policy
  for infrastructure interruption, preserving every attempted artifact.
- Primary arm S = bf16-head variant, exact GGUF/helper/client/runner/scorer
  hashes and repository commits, dynamic llama.cpp libraries/build and upstream
  executor revision, source checkpoint, tokenizer/template/config and firmware
  identities. Verify the actual artifacts at startup; saved checksums are not
  automatic enforcement of a previously approved identity.
- 8192 processed text / 2000000 chip tokens per question; helper context 16384;
  separate reference cap 2000000 machine records per call; greedy lowest-ID
  ties, EOS/forced-token rules, executor-only circuit, no context shifting.
- Exact saved original-C comparator, all paired questions, Fraction-based
  answer rules, hard-failure rules, corrected scorer and **S minus C** sign.
  Paired Newcombe method 10 with its covariance correction, 95% interval and
  fixed ±5-percentage-point margins; no difference/equivalence claim when
  neither criterion is met. Report the four paired counts and all failures.
- Freeze the intended median practical rule in S-relative form (as inherited:
  median S <= 2 × median C and <= 60 s), separately from the accuracy interval.
  Also report speed ratio and summed times; that practical ceiling by itself
  does not establish a speed improvement. If a stricter speed target is wanted,
  choose it now. Existing `--compare C,S` would orient this test incorrectly even
  after merely adding S to the scored-arm whitelist.
- Freeze the outer per-question timing boundary, startup excluded/reported
  separately, inner bucket definitions, clocks, UUID/lock, and warm-up policy.
  Apply finding 2's memory method; either re-measure original C with the same
  sampler or explicitly forgo a like-for-like memory advantage claim. Do not
  use the historical 18.98 GiB allocated peak as a process-footprint comparator.
- Freeze control gates and global design validity: zero lock mismatches,
  artifact identity unchanged, passing numerical/row/history controls, no
  missing/duplicate IDs or unexplained early ends. Helper/protocol/non-finite
  errors or OOM stop and invalidate completion; lock mismatch invalidates the
  design. Ordinary text/chip-budget exhaustion or a wrong answer remains a
  recorded failed question, not grounds for selective rerun. If memory queries
  fail, mark the measurement unavailable and withhold that claim, not accuracy.

### 4. Runner/context/lock details

The helper's context is sufficient for this registration: original C's largest
prompt is 726 tokens, so prompt plus 8192 processed text tokens fits well below
16384. Chip-only spans do not consume Qwen positions. Add an explicit
prompt-length-plus-text-budget preflight bound for future inputs rather than
assuming the model's advertised 262144 context equals the helper's allocated
context. EOS does not require another row feed. No GPU context expansion is
needed for these 160 questions.

Read the reference Session implementation: `max_records` limits complete
machine records/instructions for each call, while max_chip counts generated
chip tokens across the question. These are different units; the two million
values must not be described as identical allowances. Original C's startup
records the same separate cap, so retaining it is the fair comparison. The
reference eagerly expands the call at tool-call close, before the split consumes
its token budget; this can spend CPU time/host memory beyond the eventual
truncated chip stream. Preserve the cap and record cap exhaustion distinctly;
changing it after seeing run behavior would change the protocol.

The updated transport now has nonblocking write-all and one send-plus-reply
request deadline, plus bounded END transmission; the earlier stage-1 blocked-
send implementation is no longer present. This pass inspected that fix rather
than claiming to rerun the earlier transport suite. The new actionable runner
issue is finding 5, and the scorer blocker is finding 4.

### Final disposition

**Hold stage 3 until the control, scoring, run-validity and measurement fixes
above are complete and the revised bf16-head registration is committed.** This
is a positive end-to-end pilot with a credible split implementation, not yet a
validated full-run protocol. No new GPU jobs were used. A final read-only
nvidia-smi check showed the V100 at 13 MiB with no listed compute process;
shared 4070 processes were untouched. No main/source branch or deployment edit
is part of this review.
