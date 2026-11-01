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
