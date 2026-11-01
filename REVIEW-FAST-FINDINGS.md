# Inside-check and fast-options review

## Pass 1 — blind

Reviewed source `b8b6deb64d4060e2b395438ea0eeb441079d0351` on
`review/codex-fast`. The two results pages, second-pass brief and commit messages
after 24dc718 were not opened before committing/pushing this pass. Review only:
no fast-run build or launch, model loads, GPU jobs, installs or deployment edits.
All remote checks were read-only CPU/disk work.

### Findings

1. **NIT — print the ownership-masked ablation result alongside the all-position
   summary.** The script's summary reports 14211/14745 unchanged argmaxes, but
   that mixes Qwen-chosen predictions with positions whose next token is owned
   by the circuit. Independent reconstruction gives **13746/13746 unchanged
   Qwen-chosen picks**, and 465/999 unchanged circuit-owned picks. All 534 changed
   picks are in the latter set. Without this split a reader could mistake the
   global differences for loss of language-model invariance, or mistake all
   positions for the Qwen-owned agreement denominator. Smallest improvement:
   print both denominators, disagreement counts and top-5/gap statistics under
   the same stage-0 ownership mask.

2. **NIT — the weight scanner's norm maximum is only the last layer's value.**
   inside_check_weights.py accumulates matrix writes with max(), but assigns
   `norm = t[D0:].abs().max().item()` on each layer. An earlier nonzero norm
   extension followed by a zero last layer would still print zero for that
   weight kind. Accumulate the maximum across layers, or list exceptions by
   layer. This does not change the supplied conclusion: an independent scan of
   all 64 input/post-attention norms found zero computer-extension weights.
   Note that these are zero-centred norms: stored w=0 means gain 1, not gain 0;
   final-norm stored w=-1 is what suppresses most computer columns at the head.

### What the zeroing experiment actually establishes

The script correctly clones the saved prefill and step rows, zeros exactly
columns 2560 onward in both, resets the helper independently for each condition,
feeds prompt positions 0..P-1, then one row at P+k per saved text input. Chip
positions do not advance this text history. Thus it compares the two input
conditions on the same **teacher-forced** text history, not two free-running
answers with potentially different calls. The comparison includes the prompt's
last prediction plus all saved text-step predictions; it does not count every
prompt row as a separate argmax observation.

Recomputed the raw all-position summary exactly: 14745 positions, 14211 equal
argmaxes, 13747 equal top-5 ID lists, maximum gap difference 299.71603775 where
top-5 IDs agree. Under the Qwen-owned mask: **13746 equal argmaxes and top-5 ID
lists out of 13746**, maximum top-logit-gap difference **7.62939453125e-6**.
The large global gap differences belong to circuit-owned decisions, whose text
output-head code is deliberately changed by the ablation.

A CPU recording-helper control verifies RESET, prompt/step positions, retained
language columns and zeroed computer columns. A deliberately broken zeroing
boundary at column 0 changes the recording helper's argmax (7 to 8), so the
counterexample detects loss of the language inputs. The actual raw experiment
also has 534 changed argmaxes; it is not a probe that invariably says “same.”
This review did not perform a new GPU language-column perturbation control.

The fixed-fixture result supports input independence of the **observed
Qwen-owned choices in this mu=0 executor regime**. It is not proof of bitwise
logit equality, universal independence for arbitrary states, free-run identity,
or preserved accuracy after replacing the Qwen half with a different engine,
precision or plain checkpoint. At circuit-owned positions the executor still
needs its code/ownership logic even if the chosen language engine never sees
those dimensions.

### Weight structure and the “plain Qwen” reading

Read the permitted upstream architecture, muting and widening source. The
language matrices' original blocks are preserved; muting adds columns on mu
to DeltaNet a/b/z and attention query gates/anti-sink keys. Independent CPU
comparison with the base checkpoint found **360 language-layer tensor prefixes
bit-identical**, and **65 input/post-attention/final norm prefixes exactly equal
to the expected bf16-stored rescaling** `(1+w)*sqrt(2560/3968)-1`.

The cut checkpoint has no nonzero matrix writes into the computer extension
in these layers, and its added reads are the known mu controls. The saved
fixture count is 20486 prompt/text input rows, with **mu nonzero in zero rows**.
The text output head does retain computer features: printable-token code
columns and a common tau column, while the final norm leaves gains on code and
tau and zero gain on most other computer columns. Thus saying the whole output
head “does not use the computer dimensions” would be false. Those fields serve
ownership/output coding; the all-position ablation explicitly demonstrates it.

**“Plain Qwen up to norm rounding” is a useful structural approximation, not an
exact mathematical identity.** Both checkpoints have RMSNorm epsilon 1e-6.
For language mean-square M with zero extra residual, ideal gain compensation
changes `1/sqrt(M+eps)` into `1/sqrt(M+eps*3968/2560)` unless epsilon is also
adjusted. This difference exists in exact arithmetic, separately from bf16
rounding of the rescaled gains and later kernel arithmetic. Tiny kept residuals
can also enter the norm denominator. As a deliberately small-variance algebra
control, M=eps gives a widened/base normalization ratio about 0.8856 even with
ideal unrounded gains; this is not a measured fixture error. Typical residuals
can make the epsilon effect tiny, but “only rounding” is too literal.

The independently recomputed plain-HF agreement is **13728/13746 = 99.869%**,
18 disagreements, 12 at exact HF ties, maximum HF margin 0.25 and no margin >=1.
That supports high teacher-forced agreement on the development fixtures; it
also directly disproves bit-for-bit choice equality. It cannot identify norm
rounding as the sole cause of every mismatch or validate unseen histories.

### Speed-option reconstruction

All four benchmark files identify the V100, llama.cpp build 5e03bdd, pp512 and
tg128 at depth 0, with three timed samples per test. These are llama-bench
throughputs, not complete question latencies or executor-inclusive rates.

| GGUF | Tensor GiB | pp512 tokens/s | tg128 tokens/s |
|---|---:|---:|---:|
| widened F16, BF16 head | 12.1436 | 3581.5305 | 50.6831 |
| widened Q8_0, BF16 head | 7.3134 | 3288.7079 | 69.2235 |
| plain F16 | 8.0603 | 4974.4180 | 73.9759 |
| plain Q8_0 | 4.2837 | 4346.2044 | 98.8904 |

Q8 improves these decode rates but makes pp512 slower for each model. Faster
decode therefore does not mean every prompt/workload improves by the same ratio.
Quantize logs confirm the widened Q8 variant retains its tied head as BF16;
the plain Q8 conversion quantizes its head too. These are different precision
changes. Tensor storage is not total runtime VRAM.

The raw match files are labelled A and B8. Without the withheld results page I
retain those labels rather than infer their GGUF assignment from the names.
Independently reconstructed the stage-0 Qwen-owned decisions and reran the actual
gate on CPU:

| Raw candidate | Matches / Qwen-owned positions | Agreement | High-margin disagreement questions | Gate |
|---|---:|---:|---:|---|
| A | 13704/13746 | 99.69446% | 0 | PASS |
| B8 | 13703/13746 | 99.68718% | 0 | PASS |
| saved plain-llama control | 13718/13746 | 99.79630% | — | reference |

The fixed threshold is candidate agreement >= control minus 1 percentage point
and no more than two questions with a >=1-logit disagreement. Both pass; neither
is a full-run accuracy/equivalence result. Reset checks and the gate's data/hard
checks pass. Raw output lists do not by themselves bind their A/B8 labels to a
GGUF/helper hash; retain the generating command/identity when registering a
chosen option.

Read-only full-file hashing confirms all three GGUF hashes in gguf-sha256.txt:
qwen-half-q8_0-bf16head a3df9810..., plain Q8_0 1a04536f..., plain F16 958a023c....
No new model inference was needed to reconstruct the table or mask comparisons.

### Blind conclusion

The computer mostly communicates with the language model through forced text
and ownership/muting machinery, not learned matrix reads of arbitrary machine
state. The observed mu=0 fixture choices survive removing the extra row inputs,
while the checkpoint still has genuine computer-related control/head structure.
The near-plain structural reading is supported with the normalization and
finite-precision qualifications above. Both candidate premise checks support
further controlled development, not a prediction that end-to-end answers,
latency, or accuracy will match. Nothing for the proposed fast run was built,
launched or selected in this blind pass.

## Pass 2 — guided

Opened the second-pass brief, both results pages and FAST-PLAN.md only after
blind commit `baf76c5` was successfully pushed. The pages map candidate A to the
widened Q8_0/BF16-head file and B8 to stock Qwen Q8_0; their numbers match the
independently reconstructed gates. B is the saved plain-F16 control. No fast
implementation or inference was started.

### Additional findings

3. **REDLINE — replace exact/functionally-equivalent language with the observed
   conditional result.** The inside-check page says “the language part's choices
   depend only on the text,” “Mathematically, on text the half equals plain
   Qwen3.5-4B,” and “different model only by the norm rounding.” The plan repeats
   “functionally Qwenputer up to the norm rounding.” Pass 1's epsilon calculation
   disproves the literal mathematical identity, and the ablation establishes
   argmax invariance on fixed mu=0 histories, not every possible state or free
   trajectory. The output head also deliberately uses the code/tau fields at
   owned positions. For F specifically, Q8_0 changes learned matrices and the
   output head as well: its differences cannot be attributed only to norms.

   Smallest replacement for the page/plan rationale: “On the 12 saved mu=0
   executor histories, zeroing the computer input dimensions preserved all
   13746 Qwen-owned greedy choices. The language matrices preserve their base
   blocks, with added muting controls and norm compensation; plain HF Qwen
   agrees on 99.869% of these choices. This supports testing a stock-model
   replacement, not mathematical or free-running equivalence. Norm epsilon,
   stored-gain rounding, engine arithmetic and (for F) quantization can change
   outputs.” Keep the arms stock; do not silently alter epsilon or gains to
   force an identity claim.

   “Inside vs beside” can describe the **interface on this executor decoding
   path**: the model consumes text while the computer controls its own trace and
   output. It does not establish that every full-checkpoint execution path or
   every possible internal state has no other numerical coupling. The earlier
   135-vs-130 inconclusive result does not prove a mechanism or prove the two
   systems equal. Describe the compiled weights, executor realization and
   language interface separately.

4. **REDLINE — register a new analysis identity and F/G scoring support.**
   The build list promises “a new frozen runtime identity per arm, the same
   analysis identity.” The current scorer accepts C/L/S first, knows no F/G,
   and the stage3_analysis.py score wrapper hardcodes S,C. An isolated local
   CLI counterexample with F-labelled copies of real records is rejected before
   scoring: “the first arm must be one of ('C', 'L', 'S') (got F).” Merely adding
   F/G would then change the scorer hash and violate the old frozen analysis
   identity. The existing analysis manifest also does not freeze S as a saved
   comparator for the new secondary comparisons.

   Smallest plan edit: “Add F and G to the scorer, chain diagnostics and allowed
   first comparators, and extend the analysis wrapper to execute every
   registered comparison. Freeze a **new fast-run analysis identity** before
   either run, covering the updated scorer/parser/wrapper, C and S replies and
   question hashes, registered questions, launchers and both runtime manifests.
   Keep the old stage-3 analysis identity unchanged.” Add CLI regressions for
   F,C; G,C; F,S; G,S, signed intervals and arm-relative latency rules, plus
   unknown/missing arms, duplicate/missing IDs and existing-score regression.

### 1. Pages and speed estimate

The masked ablation, HF agreement, speed table, model sizes and stage-0 gates
are numerically supported. The normalization and scope correction is finding
3, not a reason to discard the useful structural observation. The reported
7.6e-6 top-gap difference is compatible with finite-precision/common-offset
handling; the experiment alone does not isolate that as its sole cause.

The whole-question estimate has correct **conditional Amdahl arithmetic**.
Using the exact benchmark ratio 98.890440/50.683137=1.95115 and treating the
entire S text bucket as scalable, with all other S time unchanged, gives:
4177.32/1.95115 + (5147.63-4177.32) = **3111.26 seconds**, or **1.6545x S**,
**2.4494x original C**, with the chip bucket about **28.8%** of the new time.
This is an aggregate matched-work estimate, not a prediction for every question
or a new measured result. It assumes the same token/call counts, applies
short-context decode scaling to a whole text bucket containing other work,
and omits workload changes from quantization/near-tie divergence. pp512 also
moves differently from tg128. Keep these assumptions beside the estimate;
free-run loops or changed calls can dominate it.

The table's “clear disagreements” means **questions with >=1-logit Qwen-owned
mismatches** under the registered gate, not zero disagreements overall. Candidate
A has 42 mismatches and F/B8 has 43; both have zero such high-margin questions.

### 2. Is this a sound experiment?

**Yes.** Stock Qwen in token mode beside the unchanged exact executor is a sound
new system to test, given the measured weak input dependence and near-plain
weight structure. It is appropriately framed as **“Qwenputer's exact computer
with the stock Qwen language model.”** It is not an exact-weight conversion of
Qwenputer into llama.cpp, and F is a quantized language-model variant. Circuit
ownership and reference checking, rather than an assumed identity of language
models, support the computer-correctness claim. Findings 3 and 4 should be
corrected before the build's own review/registration is considered complete.

### 3. Comparisons and selection

F/Q8_0 primary versus **C** is reasonable: C is the original-system anchor and
retains continuity with the registered port experiment. S is the right
additional comparator for the narrower “faster than the widened split”
question; keep it explicitly secondary/descriptive as planned. Choose either
anchor now if the scientific question changes, never after seeing which result
looks better. There is no need to move the primary to S merely because S had
lower observed accuracy than C.

Freeze and report G,C; F,S; and G,S regardless of their outcomes, with their
paired counts, ordinary intervals and descriptive status. “No multiplicity
correction claimed” must not become three extra confirmatory discoveries or a
reason to replace a failed F primary with G. If a gate prevents an arm from
running, report that outcome; do not promote the other arm. Clarify launcher
behavior so both arms are attempted subject to safety/preflight gates even if
one has a numerical/design failure; avoid accidentally skipping G because a
shared set-e shell exits on F. Infrastructure failures must preserve attempt
folders and follow the one disclosed restart rule.

F-then-G is an auditable fixed order and adequate for this registered system
comparison. It cannot isolate quantization from order/environment effects in
a causal F-versus-G speed claim. No timing repeatability claim follows from one
run each against saved historical comparators.

### 4. Required build and controls

Before either full run, build and review the following:

- Preserve snapshot installation and executor prompt initialization. Token
  prefill removes the need to construct external Qwen residual rows, but must
  still perform upstream ExecRow.start/reset with the same S/history outcome.
  Compare the new token path's executor state with the existing row path on
  the fixture prompts, then verify A,B,A reset through the actual runner/helper.
- Feed prompt IDs at 0..P-1 and every processed text input thereafter at P+k,
  including circuit-forced byte echoes and HALT newline. Chip HALT itself and
  other chip tokens are not language inputs. EOS is not fed. Preserve the
  existing `_exec_hc`/circuit_token ownership decision and budget/lock ordering;
  the helper's prediction must lose wherever the circuit owns the next token.
- Run the scripted parity and deliberate position/chip-feed/ownership mutations
  specified in the plan. Include text/chip-budget boundaries, wrong trace,
  given-up ownership, helper error/death/timeout, right-looking final text plus
  lock error, context preflight, complete versus incomplete run status, and
  child cleanup. Verify ordinary budget stops remain failed questions rather
  than a reason to select a new run.
- On saved fixture prompts, record actual helper token requests and compare
  them with an independently reconstructed prompt-plus-text history, not a
  log compared with itself. Include multi-call/forced-echo cases, new prompt
  resets and first-token decisions. A free-run EOS assertion is not a numerical
  helper test: retain the exact-artifact token-mode premise gate, including its
  immediate-EOS negative control, in the same session before each arm.
- Freeze per-arm GGUF/helper/libraries, new runtime code and mode/arm flags,
  exact executor/firmware, both relevant tokenizer/config/template identities
  and text-ID compatibility, questions/budgets/context, fixed arm order and
  new analysis identity. Record startup status, all attempts and timing scopes;
  use the corrected V100 UUID memory sampler through cleanup. Do not compare
  its sampled footprint with C's historical PyTorch allocation peak.

**Skipping a new full HF residual-row oracle is justified for the token-input
boundary**, provided these controls pass and the executor path is unchanged.
The specific externally constructed rows validated by stage 2 are no longer
sent to Qwen. Rows/embeddings still exist inside the stock model, so phrase the
claim as removal of that external residual interface, not absence of embeddings.
The prior same-history oracle remains supporting evidence for the unchanged
computer path; new histories are still checked token-for-token by the RV32 lock.
This does not justify skipping the actual token-history/reset checks or the
candidate's teacher-forced numerical gate.

### Disposition

The experimental direction and C-primary/F-first registration are sound after
the rationale and analysis/build omissions above are corrected. No new
teacher-forced full-HF oracle or GPU measurement is required to approve this
**plan**; the future implementation and its controls still need their promised
review before launching F or G. Neither was built or run here. No GPU job was
used, no deployment was changed, and only this review branch is being pushed.

## Build pass 1 — blind

Merged `origin/inside-check` at `2e36f0a` into the review branch. Read only the
allowed code, identity files and control outputs for this pass; neither the
build-status section nor the build pass-2 brief nor commit messages after
b8b6deb was opened. Did not execute fast.sh, including with fake commands, and
did not launch model inference. CPU checks and gateway reads only.

### New findings

5. **REDLINE — analysis ignores arm validity when all rows exist.**
   The registration distinguishes invalid-design/incomplete arms from completed
   measurements. fast_analysis.py verifies frozen analysis code and input
   checksums, then only checks whether replies.jsonl exists before scoring; it
   never reads the arm's startup status or rejects any lock error globally.
   A lock error on the last question leaves all 160 rows present, so ordinary
   completeness checking cannot enforce the design-validity rule.

   CPU reproduction: use genuine stage-3 records relabelled F/G in owned
   temporary folders, put a trace mismatch and inside.lock_error on F's last
   question, and write F startup status “invalid design.” The frozen analysis
   verifier passes; all four comparisons exit 0; the wrapper exits 0 and emits
   a primary F-vs-C verdict (131 becomes 130 correct), rather than withholding
   it. This is a synthetic analysis test, not new F/G experimental data. A
   nonzero helper exit after 160 successful answers has the same completeness
   problem if startup marks the arm incomplete.

   Smallest fix: validate each new arm before invoking its comparisons: require
   its startup status ok, correct arm/token mode and frozen runtime/GGUF identity,
   zero row lock errors/mismatch records/harness errors, and the registered
   completeness checks. Emit an explicit withheld/not-run outcome for invalid
   or incomplete arms, preserve their data, still report eligible G comparisons,
   and return a nonzero overall status without promoting G to primary. Ordinary
   budget failures remain scored failed questions and do not invalidate the
   whole arm. Add wrapper-level regressions for a final-question lock error,
   failed helper shutdown with all rows, missing startup and a valid control;
   refresh the frozen analysis identity after the fix.

6. **NIT — independent history reconstruction rejects a legal partial run.**
   test_split_tokens_gpu.py permits text-budget stops, but reconstruct() insists
   that a tool call's entire reference-generated run is already present in gen.
   CPU counterexample: run the real scripted token path with max_new=27, exactly
   through the first </tool_call>. It stops with text_tokens=27 and no lock
   error, before appending GO. This is upstream's valid budget behavior, but
   reconstruct() raises “the run after 26 differs from the reference's.” A
   text-budget stop in a forced echo can have the same issue.

   Smallest fix: pass explicit stop/count information to reconstruction and
   accept only a correctly matching prefix of the reference run at a registered
   budget boundary; return the text inputs actually processed before that cut.
   Continue rejecting missing/mutated tokens in a completed EOS run. Add cutoff
   controls at tool-close, inside a chip span and inside a forced echo. The saved
   F budget-stop fixtures end outside such a partial run and pass; this does
   not invalidate their recorded controls or the production loop.

7. **NIT — fail immediately if attempt-directory creation fails.**
   fast.sh checks existence, then ignores failure of `mkdir "$OUT"` and
   `mkdir "$A"`. In particular, a directory created between check and mkdir
   could let the script continue and truncate an existing arm.out via `>`.
   This is defensive preservation of attempts, not evidence of an actual
   overwrite. Use an atomic successful mkdir as the gate (`mkdir ... || exit`),
   and never proceed into an arm directory whose creation failed. Syntax was
   checked; the launcher itself was not run.

### Token path and controls

The production delta is small and correct: after snapshot and helper.reset,
token mode calls unchanged `_exec_start(ids)` before sending prompt IDs at
0..P-1. On each processed text input it sends exactly nxt at pos, advances pos
once, then lets circuit_token override the helper. Chip inputs advance only
the executor; EOS is checked before either token feed or executor step. Forced
byte echoes and the HALT newline remain text inputs. `_exec_hc` is retained for
ownership; the original row path is preserved. Lock and budget ordering have
not moved. Main's token-mode warm-up is followed by the per-question reset.

Ran the hash-matched deployed CPU test_split_chat.py with CUDA hidden: row and
token modes agree with upstream on one/two calls, both budgets, bad chip trace
and given-up ownership. All three deliberate mutations are caught: skipped
positions, helper overriding circuit ownership, and an extra feed during a
chip step. This tests actual production split_generate using scripted model/
executor choices, not loaded model weights.

The GPU checker reconstructs the computer's portion independently: it gives
the reference only the model's free text, recomputes each run, validates that
against the generated chip/echo span, then builds expected language requests.
It is materially stronger than just filtering its recorded requests or gen.
A separate CPU reproduction on a real reference call accepts the complete
stream and rejects a corrupted forced HALT newline. Its budget-prefix limit
is finding 6. Initial executor S/history comparison and the final A,B,...,A
check support prompt initialization/reset. Numerical helper validity remains
the separate exact-artifact preflight gate, not the mere fact a run reaches EOS.

The allowed gateway log tokens-gpu.out records an **earlier F control failure**:
its old harness required EOS, so two valid text-budget stops were marked BAD;
G then completed all fixtures successfully. The later tokens-gpu-F.out and
checked-in F log use the corrected stop criterion and pass, including those
two 8192-text stops. The request counts agree between attempts. This is a
visible development control correction, not a registered 160-question rerun.
The checked-in G log uses the older EOS wording, but all its cases ended EOS,
so the later broadened criterion does not weaken any observed G result.
No new GPU run was necessary in this review.

### Scorer, identities and launcher

- Ran test_score_fast.py locally: all four 160-row comparison checks, signed
  differences, unknown/missing-arm handling, duplicate/missing-ID rejection,
  and exact historical stage-3/stage-1/inside-4b score-object regression pass.
  The relabelled test inputs are explicitly synthetic and prove wiring, not
  experimental accuracy of F/G. F/G are now in scoring, chain diagnostics and
  first-arm support; this closes the earlier missing-arm implementation gap.
- fast_analysis.py verify passes for all **12 analysis files**, whose hashes
  also match its pinned commit 10899f3752006ec470dd689add358be7068ec951. It
  freezes the new scorer/parser/wrapper, C/S comparator bytes and question
  hashes, question file, launcher/builder and both runtime manifests. That
  closes the earlier stale-analysis-identity omission. It does not freeze
  future F/G outcomes, and currently does not validate their run status or
  provenance before scoring (finding 5).
- Both deployed runtime manifests match the committed bytes and every listed
  current file: **110 files per arm, 111 unique files** in their union, with
  only the selected GGUF differing. They bind helper/libraries, new token loop
  and runner, launcher, executor/firmware, prompt and tokenizer/config assets,
  gate inputs and the per-arm stock-model file. Both record text-ID
  compatibility, token mode and the correct arm. The checker verifies listed
  bytes; the frozen launcher supplies the matching CLI choices. Arbitrary
  manual CLI arguments are not automatically validated against manifest
  metadata.
- By source inspection, F's set-e/pipefail scope is confined to its subshell.
  Parent status capture records a nonzero F exit and continues to G; a sticky
  rc makes the overall launcher nonzero if either fails. Each arm has its own
  preflight, helper log, run folder and arm.out, and the top-level root refuses
  an existing path. The directory-creation edge is finding 7. No launcher test
  required actually running fast.sh; it was not executed.

### Build-pass disposition

Token-mode execution and the recorded controls support the intended build,
with the secondary control/preservation nits above. **Fix the analysis validity
check and refreeze its identity before the registered fast run.** A complete
row count must not turn an invalid design into a primary accuracy verdict.
Nothing was built or launched by this review; only findings on this review
branch are being committed and pushed.

## Build pass 2 — guided

Opened REVIEW-FAST-BUILD-PASS2.md and FAST-PLAN.md v2/build status only after
build blind commit `ef5f988` was successfully pushed. Did not run fast.sh.

### 1. Development behavior and preserving registration

**Keep F primary and G secondary.** Selecting G now because its development
answers terminate more readily would change the primary after seeing relevant
behavior. F's two budget stops are legitimate failed answers, not lock failures;
they must remain in the registered denominator. The original EOS-only GPU
control was stricter than the plan's legitimate budget-stop rule. Correcting
that harness and retaining the earlier failed log is appropriate development
work, not a reason to discard those model outcomes.

The observation is usefully disclosed in the build status. The logs substantiate
F's 8192-text stops on final-2-003 and final-4-001, no call on the latter, and
4496 versus 1477 text requests on final-2-001. They make clear why a faster
short-context tg128 rate need not mean a faster complete answer: F can do much
more work. These development prompts overlap the registered benchmark; do not
call the full 160 an untouched held-out set or treat these controls as an
independent accuracy experiment.

For the full runs, retain per-question processed text/chip counts, calls,
stop reasons, raw outputs and outer/bucket times, and report text-budget counts
and their time contribution. Existing scorer loop flags detect repeated tool
expressions; they do not reliably identify a no-call textual reasoning loop.
Label them tool-call repetition flags. A descriptive textual repetition measure
could be added and frozen now if wanted, but it must not change answer grading,
become a new adaptive abort rule, or justify selective reruns. No extra model
measurement is required just to decide whether to preserve the primary.

If F's premise gate fails, report that the registered primary did not run.
If it has a design or infrastructure failure, use the corresponding invalid/
incomplete classification. Eligible G results can still be reported as
secondary; they never substitute for F. This is why finding 5's wrapper validity
gate is necessary even though launcher and runner already exit nonzero.

### 2. Interpreting G versus stage 1

G uses the same plain-F16 GGUF as L, with the same prompt/token conventions and
an exact calculator response interface. It is therefore reasonable to describe
G as close to **stage 1's language path with executor-produced computer traces
instead of bulk reference emulation**, subject to the actual budget/history and
runtime checks. It is not another independent demonstration that the widened
Qwenputer language half was ported unchanged.

I independently verified that G's text-request and call counts match saved L
on all 12 fixture IDs. The supplied GPU-control logs do not persist generated
texts or token hashes, and test_split_tokens_gpu.py does not compare them with
L, so the “word-for-word” development assertion cannot be reconstructed from
these logs alone. Preserve a per-fixture token/text hash comparison if that
stronger claim is used. This is a reporting limitation, not a new execution
blocker; the registered G replies will allow a direct complete comparison with
saved L without changing the primary protocol.

Identical language outputs would not imply identical wall time. L computes the
reference result in bulk, while G steps the compiled circuit and checks its
trace. G-versus-C remains the registered secondary comparison of two complete
systems with different language numerics/weights and execution arrangements;
it is not an isolated test of circuit placement or a fresh base-model accuracy
sample. F adds language-weight quantization and can change trajectories much
more, as the development counts illustrate. F/G-versus-S comparisons remain
descriptive, with historical-comparator and fixed-order caveats.

### 3. Readiness

The updated plan fixes the earlier unsupported exact-equivalence rationale and
now explicitly describes stock/quantized models, finite-precision differences
and a new analysis freeze. F/G scoring, the four comparisons and frozen C/S
inputs are implemented and verified. The production token feed/ownership path
and recorded development controls support this build. A new full HF residual-
row oracle is still unnecessary at this token-only interface; the actual-token
and reference checks remain necessary and are present.

**Do not launch the registered runs yet:** fix build finding 5 so a complete but
invalid/incomplete arm cannot receive an acceptance/primary verdict. Add the
wrapper-level negative controls and refresh fast-analysis.json before data.
A focused CPU re-check is sufficient for that analysis fix; no new GPU
experiment is needed to establish it. If taking the optional fast.sh mkdir fix,
refresh both runtime manifests and then the analysis manifest as well, because
the launcher is hashed in each. The partial-budget reconstruction fix can be
validated with the CPU boundary cases; it does not require reinterpreting the
existing successful fixture histories.

Do not change F/G selection or the registered answer/budget rules while making
these plumbing corrections. The only required release hold is the analysis
validity guard; the two build-pass nits and the G-text provenance limitation
are recorded separately. After that guard passes its negative controls and the
new freeze is verified, the registered per-arm live preflights must still pass
when the operator starts the run.

### Review completion

Both build passes are complete. The review changed only this findings file on
the review branch. No registered fast run, model load, GPU job, deployment edit,
source/main merge or external publication was performed. fast.sh was inspected
and syntax-checked, never executed.

## Confirm

Merged `origin/inside-check` at `96c22f9` into this review branch and followed
REVIEW-FAST-CONFIRM.md. Re-ran the independent counterexamples and mutations
before reading the requested 2e36f0a..96c22f9 registration/identity diff.
CPU only; fast.sh was never executed.

### Findings 5–7: closed

**5 — CLOSED: invalid/incomplete arms no longer receive comparison verdicts.**
Replayed the original final-question lock-error case with all 160 rows present,
correct frozen arm/mode/GGUF metadata, and F status “invalid design.” The actual
verify-then-score CLI exits **4**, emits F's withheld validity result, creates
neither F-vs-C nor F-vs-S score output, and still creates both eligible G
comparisons. G is not promoted to primary. The original counterexample no
longer produces an F primary verdict.

Ran all six test_fast_analysis.py controls: both valid arms score all four
comparisons; a final-row lock error, incomplete helper shutdown with all rows,
missing startup and wrong GGUF each withhold F while scoring G; ordinary text
budget stops do not invalidate an otherwise completed arm. The new gate checks
row lock/mismatch/harness failures independently of startup status, so a
right-looking status cannot hide the final-row trace mismatch. Existing scorer
completeness and question-identity checks remain in the comparison path.

Independent in-memory mutation: replaced arm_problems with an always-valid
answer. **Four negative controls fail and the control suite exits 1.** This
proves the new gate tests detect the original regression, rather than merely
observing successful scorer output. All test records were temporary copies of
real data with synthetic labels/status; no F/G measurements were generated.

**6 — CLOSED: legal budget prefixes are accepted without accepting corruption.**
Replayed the exact original max_new=27 token-mode case: the stream ends at
</tool_call>, text_tokens=27, no lock error. Reconstruction now accepts it with
the explicit budget flag and returns precisely its processed text inputs.
The new controls pass for complete EOS, cuts at tool close, within a chip span,
and immediately after a forced echo. They reject the same truncations without
a budget stop, changed chip/forced-echo tokens, and a missing HALT newline.

Independent mutation: removed the condition requiring budget_stop for a partial
reference run. The **three unregistered-truncation checks fail and the suite
exits 1**. The real code still checks every available prefix token against the
reference; returning text from the validated input stream does not make that
computer-span check circular. The GPU caller's final stop check separately
requires a permitted stop, so merely passing the reconstruction flag does not
accept an unexplained early end.

**7 — CLOSED: directory creation is now the atomic preservation gate.**
The top-level mkdir failure exits before any arm/log writes. An arm mkdir
failure records failure and continues to the other arm without redirecting into
the existing directory. Tested the two exact guard statements in isolated local
shell snippets: pre-existing/raced root and F directories retain a sentinel
arm.out unchanged; the isolated G guard can still succeed and overall status
remains nonzero. This executed only the guards, **not fast.sh or its job body**.
The full launcher also passes bash -n syntax checking.

### Frozen identities and wording

- Each runtime manifest still lists 110 files. Compared with 2e36f0a, **only
  /mnt/nvme/flashnext-test/llamacpp/fast.sh changes hash** in either file;
  all other entries and arm/GGUF/mode metadata are identical.
- Independently hashed all listed current gateway runtime files (111 unique
  files across the two arms): **zero mismatches**. Deployed manifest hashes
  match the committed versions, F `ff3a631b...`, G `bd5daae8...`.
- The new **12-file analysis identity verifies**, and every listed byte hash
  matches its pinned commit `0f31f2662802619453118aadb1c5d90db2ef1a51`.
  The previous identity is preserved exactly as
  fast-analysis-v1-superseded.json. The comparisons/roles and scientific
  registration have not been switched in response to development behavior.
- FAST-PLAN now accurately claims matching G/L request and call counts, not
  unpreserved identical texts. It labels loop flags as tool-call repetition,
  adds no new adaptive abort rule, and states the analysis validity/withholding
  policy now implemented. Raw outcomes, token counts, stops and timings remain
  required for both arms.

### Readiness

**The hold is cleared: fast.sh may run as registered**, with F primary, G
secondary, both attempted in their fixed order subject to their own live
identity/preflight gates. Invalid/incomplete F must remain withheld; eligible
G stays secondary. There is no remaining release blocker from findings 5–7,
and these CPU-only fixes do not require another development GPU run.

This is a readiness verdict, not a launch: no registered run, GPU/model load,
deployment edit or source/main merge was performed. Only the confirmation in
this review branch is committed and pushed.

## Results pass 1 — blind

Merged `origin/inside-check` at `b5e7540` and followed REVIEW-FAST-RESULTS.md.
The results narrative, results pass-2 brief and commit messages after 96c22f9
were not opened before this pass was committed/pushed. No source changes,
registered runs, GPU jobs or model loads were performed by this review.

### Finding

No blocker or redline against the saved runs' registered results.

8. **NIT — disclose the initial sampling gaps, not just zero unavailable
   samples.** F has a **3.503-second** gap between its initial after-load sample
   and the first periodic sample; G has **35.663 seconds**. These precede the
   question loop, during startup hashing, as in the stage-3 sampler behavior
   already reviewed. After periodic sampling starts, maximum gaps are 0.505 s
   for F and 0.506 s for G. Zero unavailable queries does not imply no sampling
   gaps or continuous-peak coverage. Report the figures as observed sampled
   footprints and disclose the startup gaps. No primary-answer/timing
   invalidation follows: these are not pauses inside question generation.

### Registration/provenance audit

- Runtime and analysis source is unchanged from the confirmed build. All
  **110 files in each runtime manifest** (111 unique files) currently hash
  correctly on the gateway; deployed manifest bytes match the committed ones.
  All **12 analysis files** verify and match pinned commit 0f31f2662802.
- Freeze metadata places the analysis/source at 23:07:19/23:07:34 Chicago,
  before the logged F identity check at 23:20:31. Only commit metadata was read.
  The frozen scorer/parser, C/S comparators, model files and launchers have not
  been silently replaced for this analysis.
- One recorded V100 session locks clocks at 1380 MHz, performs F then G, records
  exit 0 for each, and restores clocks/releases at the end. Each arm records
  110 matched artifacts and its own token-mode preflight PASS before generation.
  CPU re-evaluation of the actual gateway preflight replies passes both gates:
  F 99.68718% versus control 99.79630%, G 99.79630% versus that control,
  13746 Qwen-owned positions, zero high-margin disagreement questions each.
- Both arms have 160 unique rows in the registered order. All prompt-ID hashes
  match C; question checksums match the registered 160-question file. Startup
  records the intended arm, token mode, exact GGUF/helper hashes, 8192-text /
  2000000-chip budgets, the separate 2000000 reference-record cap, context 16384,
  status ok and no reason/error. Maximum prompt+processed-text length is 8918.
- The gateway raw replies, startup, memory samples, arm logs, gate logs and
  question hashes match their committed uncompressed bytes. The visible
  attempt inventory has one fast-run root with F/G. Both logs contain exactly
  160 progress entries and completion. No restart, selective substitution or
  hidden per-question rerun appears in this evidence; this is not a claim to
  prove the absence of every unrecorded action.
- Re-ran the frozen verify-then-score wrapper on owned local copies. Both arm
  validity objects and all four **complete score objects (summary and rows)**
  reproduce exactly. The wrapper's validity checks accept the actual metadata
  and find no row-level lock/mismatch/harness error.

### Independent accuracy/interval reconstruction

Independently parsed each last ANSWER line, compared exact Fraction values,
and applied transport/harness, unfinished/budget/open-call and trace-mismatch
hard failures. **Every correctness decision agrees with the scorer** in all
four comparisons (all 640 distinct F/G/C/S records; L also grades 129/160).
No answer-classifier source was imported for this independent reconstruction.

Pairs below are both right / first only / comparator only / both wrong.

| Comparison | Pairs | First minus comparator | Independent 95% CI, percentage points | Result |
|---|---|---:|---:|---|
| **F−C, primary** | 123 / 8 / 12 / 17 | -2.50 pp | [-8.28823, +3.20706] | inconclusive |
| G−C, secondary | 120 / 9 / 15 / 16 | -3.75 pp | [-10.00588, +2.42359] | inconclusive |
| F−S, secondary | 121 / 10 / 10 / 19 | 0.00 pp | [-5.74519, +5.74519] | inconclusive |
| G−S, secondary | 129 / 0 / 2 / 29 | -1.25 pp | [-3.87800, +1.22080] | within ±5 pp |

Used paired Newcombe method 10 with the registered positive-covariance
correction, evaluated independently. All four first-arm latency ceilings pass.
G−S's ordinary interval lies within ±5 points, but that is the registered
**descriptive secondary** comparison, without a multiplicity correction. It
must not replace F−C as the confirmatory result. Equal F/S totals are not
identical outcomes and do not establish equivalence: there are 20 discordant
pairs and their interval extends outside ±5 points. Neither F nor G has
established equivalence/noninferiority to C within the registered margin.

### Recomputed totals

| Quantity | F (Q8_0) | G (f16) | Saved S | Saved C |
|---|---:|---:|---:|---:|
| Correct | 131/160 | 129/160 | 131/160 | 135/160 |
| Groups 1 / 2 / 3 / 4 correct | 69 / 9 / 34 / 19 | 69 / 10 / 33 / 17 | 69 / 10 / 34 / 18 | 69 / 10 / 32 / 24 |
| Median outer seconds | 7.190 | 7.765 | 10.005 | 16.770 |
| Sum outer seconds | 3175.92 | 3966.98 | 5147.63 | 7620.77 |
| Text tokens | 203096 | 206509 | 200331 | 182707 |
| Checked chip tokens | 1447753 | 1554614 | 1552141 | 1615594 |
| Text bucket seconds | 2274.15 | 3005.65 | 4177.32 | 6438.53 |
| Chip bucket seconds | 835.64 | 896.20 | 897.06 | 943.65 |
| Text tokens / bucket s | 89.3063 | 68.7069 | 47.9568 | 28.3771 |
| Chip tokens / bucket s | 1732.5080 | 1734.6731 | 1730.2533 | 1712.0691 |
| EOS / text-budget stops | 152 / 8 | 154 / 6 | 156 / 4 | 156 / 4 |
| Budget-stop question time | 805.26 s | 765.11 s | 700.96 s | 1342.18 s |
| Budget-stop share of total | 25.3552% | 19.2870% | 13.6171% | 17.6121% |

Observed comparator/first speed ratios:

| First vs comparator | Median | Summed time |
|---|---:|---:|
| F vs C | 2.33241x | 2.39955x |
| G vs C | 2.15969x | 1.92105x |
| F vs S | 1.39152x | 1.62083x |
| G vs S | 1.28847x | 1.29762x |

These describe the single registered runs and historical comparators, not
repeatability or an engine-only causal effect. Bucket rates include their
registered per-text/per-chip work and are not whole-question throughput.

### Errors, budgets and timing consistency

Both F and G have zero lock errors, harness errors or open calls. Neither
reaches its chip budget. Their budget failures are retained as failures, not
removed or rerun:

- F: final-1-060; final-2-003; final-3-015/027; final-4-001/007/032/039.
  Three budget cases (4-001/007/032) make no tool call. The longest F question
  is 4-039 (117.25 s, 8192 text, 41688 chip tokens).
- G: final-1-060; final-4-006/007/019/027/029. The longest is 4-006
  (150.77 s, 8192 text, 52502 chip tokens).

Each arm has one scorer tool-call repetition flag; this is not a complete
textual-loop detector. A no-call budget stop is not a circuit execution error.
The scorer's backend-labelled failures (three F, two G) contain explicit
`error (too long: split it)` tool responses; those diagnostic labels are not
hidden CUDA/helper crashes or lock mismatches. Helper stderr contains no
error/failed/non-finite/OOM entries. Its zero/infinite internal eval counters
remain unsuitable as independent timing measurements.

After accounting for the midnight rollover, adjacent completion times reconcile
with question seconds within the log's one-second precision (F residual range
-0.98 to +0.98 s, G -0.88 to +0.89 s). No extra multi-second inter-question
pause appears. No unexplained early termination or sampled-query failure was
found. Finding 8 concerns startup only.

### Memory reconstruction

| Quantity | F | G |
|---|---:|---:|
| Samples | 6355 | 7940 |
| Unavailable queries | 0 | 0 |
| Run / shutdown samples | 6354 / 1 | 7937 / 3 |
| After-load sum | 8896 MiB | 12618 MiB |
| Max simultaneous sum | 9070 MiB (8.8574 GiB) | 12784 MiB (12.4844 GiB) |
| Parent / helper maxima | 1708 / 7362 MiB | 1710 / 11074 MiB |

All per-sample sums are consistent; the UUID is the registered V100. Both final
samples show the helper absent during shutdown and the parent still present.
Median actual interval is 0.5 s, with the startup exceptions in finding 8.
Relative to S's same-method observed 17254 MiB maximum, these sampled totals
are lower by 47.43% and 25.91%, respectively. They are not continuous peaks,
and C's historical PyTorch allocation figure is not a valid comparator.

### G versus L and interpretation

**All 160 G full generated texts match L exactly**, as do text counts, answer
outcomes and six budget-stop IDs. The earlier development count-only claim can
now be strengthened using actual registered reply bytes. This does not imply
identical chip instruction streams: G records 26 more chip tokens in aggregate
than L's reference-trace counter, spread over four questions; text identity
alone is not a trace-identity proof.

G is slower than L on this matched language output: G's 3966.98 seconds versus
L's 2839.78, and median 7.765 versus 4.665. This is consistent with stepping
and checking a compiled computer rather than producing a result by bulk
reference emulation; the comparison does not isolate every source of overhead.
The fast system should not be described as faster than every available
computer-beside baseline simply because it beats C or S.

The registered primary F−C accuracy result remains inconclusive, while F shows
a substantial observed speed and sampled-footprint improvement. Its eight
budget failures consumed a quarter of total time. Both computer traces remain
reference-checked, but correct computer execution cannot make the model call
the computer or reason correctly. These are stock-language-model-plus-executor
systems, not native llama.cpp execution of the circuit or exact-weight
Qwenputer conversions. Development selection, fixed F-then-G order, changing
trajectories and single historical runs limit causal/generalized claims.

No new measurement or GPU job was needed for this reconstruction. The recorded
session reports clock restoration and release; this review did not acquire a
GPU lock or stop any shared process.
