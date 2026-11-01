# Qwenputer's computer with the stock Qwen in llama.cpp (plan v2, 2026-10-31; Codex review baf76c5 / 739fda6 applied)

## What you'll have at the end
The same exact computer (upstream's fp32 executor, every chip token checked against the RV32 reference) with the
**stock Qwen3.5-4B in llama.cpp** as the language model: no widened layers, no conversion of Qwenputer's checkpoint.
And a measured answer: how fast, and at what accuracy, against the original Qwenputer (C) and stage 3's split (S).

## Why this is worth testing (from the inside-check, `results/inside-check.md`)
On the 12 saved mu=0 executor histories, zeroing the computer input dimensions preserved all 13,746 Qwen-owned greedy
choices. The language matrices preserve their base blocks, with added muting controls and norm compensation, and
plain HF Qwen agrees on 99.869% of these choices. This supports testing a stock-model replacement, not mathematical
or free-running equivalence: norm epsilon, stored-gain rounding, engine arithmetic and (for F) quantization can
change outputs. The arms stay stock: no adjusted epsilon or gains to force an identity. The computer's correctness
rests on its own ownership and reference checking, not on an assumed identity of language models.

## The arms (all three registered now; no picking a winner afterwards)
| Arm | Language model | Status |
|---|---|---|
| **F (primary)** | stock Qwen3.5-4B **Q8_0** GGUF (`qwen35-4b-q8_0.gguf`, from the f16 GGUF with llama-quantize, default Q8_0 tensors) | new run |
| G (secondary) | stock Qwen3.5-4B **f16** GGUF (the stage-1 file) | new run |
| S, C | stage 3's split; inside-4b's original | saved, not re-run |

Development evidence for choosing these, teacher-forced on the 12 fixtures (`results/speed-options.md`):

| Arm | Agreement with Qwenputer's picks | Gate | tg128 |
|---|---|---|---|
| F | 99.687% | PASS | 98.9 tokens/s |
| G | 99.796% | PASS | 74.0 tokens/s |
| S | 99.782% | PASS | 50.7 tokens/s |

## What gets built (before any run; reviewed again)
- **split_generate with a token-mode language model:**
  - The snapshot installation and the executor's prompt initialization (ExecRow.start: zero state and history,
    the prompt, load_init) stay exactly as now. Token prefill only removes the external rows. The token path's
    executor state is compared with the row path's on the fixture prompts.
  - The prompt ids go in at 0..P-1, then every processed text input at P+k, forced byte echoes and the HALT newline
    included. Chip tokens (HALT itself included) aren't language inputs, and EOS isn't fed.
  - `_exec_hc` / `circuit_token` ownership, the budgets and the lock order are unchanged; the helper's prediction
    loses wherever the circuit owns the next token.
- **`run_split.py --lm-input tokens --arm F|G`.**
- **The scorer and analysis:**
  - Add F and G to `stage5/score.py` (scoring, chain diagnostics, allowed first comparators). Extend the analysis
    wrapper to run every registered comparison (F,C; G,C; F,S; G,S).
  - Freeze a **new fast-run analysis identity** before either run: the updated scorer, parser and wrapper; C's and
    S's replies and question hashes; the questions; the launchers; and both runtime manifests. Stage 3's analysis
    identity stays unchanged.
  - CLI regressions: F,C; G,C; F,S; G,S with signed intervals and arm-relative latency rules; unknown or missing
    arms; duplicate or missing ids; existing scores re-scoring identically.
- **CPU controls:**
  - Token mode equals rows mode on scripted fakes: the same tokens, counts and lock errors. Also covered: the
    text/chip budget edges, a wrong trace, the circuit giving up, helper error/death/timeout, right-looking text with
    a lock error, the context preflight, complete vs incomplete status, and child cleanup.
  - Ordinary budget stops stay failed questions.
  - Mutations: a skipped position, a chip token sent, the helper's choice overriding the circuit.
- **GPU controls on the fixture prompts:**
  - Record the helper's actual token requests and compare them with an **independently reconstructed**
    prompt-plus-text history: multi-call and forced-echo cases, new-prompt resets, first-token decisions. The
    comparison isn't a log checked against itself.
  - A, B, A reset through the real runner and helper.
  - The exact-artifact token-mode premise gate (stage-0 gate, with its immediate-EOS negative control) in the same
    session before each arm.
  - A new teacher-forced HF row oracle isn't needed for token input, provided these pass and the executor path is
    unchanged (Codex).
- **Frozen per arm:** the GGUF, helper and libraries, the new runtime code and mode/arm flags, the exact
    executor/firmware, both tokenizer/config/template identities and text-id compatibility, the questions, budgets
    and context, and the fixed arm order.

## Build status (2026-10-31, before any registered run)
- **Built:** token-mode `split_generate` and `run_split.py --lm-input tokens --arm F|G`, arms F and G in the scorer,
  `llamacpp/fast_analysis.py` (the four registered comparisons), `llamacpp/make_fast_artifacts.py`,
  `llamacpp/fast.sh`.
- **Controls:**
  - `test_split_chat.py` 28 CPU controls, including token-mode parity with upstream and three mutations;
  - `test_run_split.py` 8;
  - `test_score_fast.py` 11 (F,C, G,C, F,S, G,S; refusals; earlier scores re-score identically);
  - `test_split_tokens_gpu.py` 13 per arm on the V100, all passing (`results/fast/controls/`): the executor state
    equals the row path's, there are no lock errors, the helper's token requests equal the history the RV32
    reference reconstructs on its own, and A, B, ..., A holds.
- **Frozen:**
  - runtime identities `results/fast/fast-artifacts-F.json` and `-G.json` (110 files each, text-id compatibility
    checked);
  - analysis identity `results/fast/fast-analysis.json`.
  Stage 3's analysis identity is unchanged, but the scorer file has changed since, so stage 3 re-scores from its
  pinned commit (a335797). `test_score_fast.py` also shows the current scorer re-scores stage 3 identically.
- **A development observation, after the arms were registered:**
  - On the 12 fixture questions, F (Q8_0) hit the 8,192-token text budget on 2 (final-2-003, final-4-001; the
    second with no call), and its answers ran longer elsewhere too (final-2-001: 4,496 vs 1,477 text tokens).
  - G (f16) ended EOS on all 12, with the same text-request and call counts as stage 1's plain run (L) on every
    question. The control didn't save the texts, so "identical texts" isn't claimed; the registered G replies will
    allow the direct comparison with L.
  - This doesn't change the registration: F stays primary, G secondary, and both are run and reported. Choosing
    the primary arm again after seeing these runs would be the post-hoc selection the registration rules out.

## The registration (as stage 3, except where noted)
- **Questions:** inside-4b's 160 (`e3f98862...`), in order; one run of F, then one of G, in that fixed order; no
  reruns or substitutions. **Both arms are attempted**, each subject to its own preflight: a gate failure,
  numerical or design failure in F doesn't skip G (the launcher doesn't share a set-e exit between arms), and every
  attempt folder is kept. F-then-G is an auditable order, not a causal F-vs-G speed test.
- **Budgets:** 8,192 text and 2,000,000 chip tokens per question, the reference's separate cap of 2,000,000 records
  per call, helper context 16,384; greedy, the lowest id at a tie.
- **Preflight:** per arm, its GGUF and helper must pass the stage-0 gate (token mode) in the same session.
- **Primary comparison: F minus C**, as stage 3 (Newcombe method 10, 95%, ±5-point margins, inconclusive unless
  the interval says otherwise; the latency rule median F <= 2 x median C and <= 60 s).
- **Secondary, reported regardless of outcome:** G minus C, F minus S, G minus S (the same method, their paired counts
  and ordinary intervals, labelled descriptive). No multiplicity correction is claimed, so they aren't extra
  confirmatory findings, and G doesn't replace a failed F primary. If a gate stops an arm, that's reported; the
  other arm isn't promoted. The speed ratios (median and summed) of F and G vs S and C are reported; one run each
  gives no repeatability claim.
- **Validity:** any lock error invalidates that arm (its run stops); an exception or OOM stops it as incomplete; one
  disclosed restart from the beginning per arm for infrastructure interruption. Memory by stage 3's method; no
  comparison with C's PyTorch figure.
- **Reported for every arm:** per-question text and chip counts, calls, stop reasons, raw outputs and outer/bucket
  times; the text-budget count and its share of the time. The scorer's loop flags are **tool-call repetition
  flags** (repeated tool expressions), not a detector of textual loops; no new loop measure or abort rule is added.
- **Scoring validity (Codex build review, finding 5):** `fast_analysis.py` scores an arm only if its run is valid
  (status ok, the right arm and mode, the frozen GGUF, every artifact verified, no lock error, trace mismatch or
  harness error). Otherwise its comparisons are withheld and reported as such, and the other arm's are still
  scored, never promoted.
- **Framing:** F and G are "Qwenputer's exact computer with the stock Qwen language model": a new system, not an
  exact-weight conversion of Qwenputer into llama.cpp, and F's language model is quantized.

## Stop rules
- A preflight gate failure for an arm: that arm doesn't run; report.
- A lock error: the arm is invalid; report; no fix-and-rerun inside this registration.
