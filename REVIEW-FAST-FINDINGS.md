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
