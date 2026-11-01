# Qwenputer's computer with the stock Qwen in llama.cpp (plan v1, 2026-10-31; for Codex's review, then Chris's yes)

## What you'll have at the end
The same exact computer (upstream's fp32 executor, every chip token checked against the RV32 reference) with the
**stock Qwen3.5-4B in llama.cpp** as the language model: no widened layers, no conversion of Qwenputer's checkpoint.
And a measured answer: how fast, and at what accuracy, against the original Qwenputer (C) and stage 3's split (S).

## Why this is legitimate (from the inside-check, `results/inside-check.md`)
- **What the language half sees:** Qwenputer's 32 Qwen layers write none of the computer's dims and read only `mu`,
  the mute marker. `mu` was 0 in every row Qwen received on the fixtures (it's 1 only on chip tokens, which skip
  Qwen's layers in the chat loop).
- **The zeroing test:** with every computer dim zeroed, Qwen's choices are unchanged at all 13,746 Qwen-chosen
  fixture positions.
- **Plain Qwen:** Qwenputer's HF choices equal plain Qwen3.5-4B's on 99.869% of those positions, with only
  near-ties differing. On text, the half equals plain Qwen up to the rounding of its rescaled norms.

So feeding plain Qwen the text tokens gives the language model everything Qwenputer's own half uses. The computer's
choices still come from the executor and still win where the circuit owns the next token.

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
- `split_generate` with a token-mode language model: the prompt as token ids at 0..P-1, then each text input's id at
  P, P+1, ... (forced echoes and the HALT newline included, as now). The executor steps, `_exec_hc` (for
  `circuit_token`), ownership, the lock, budgets and EOS are unchanged; only what goes to the helper changes.
- `run_split.py --lm-input tokens`, the preflight gate in token mode (`run_helper_fixtures.py --mode plain` with the
  arm's GGUF), a new frozen runtime identity per arm, the same analysis identity.
- **CPU controls:** token mode = rows mode on scripted fakes (the same tokens, counts and lock errors); the tokens
  sent are exactly the text inputs in order, at P, P+1, ...; no chip token is sent; a circuit-owned token still wins.
  Negative controls: a skipped position, a chip token sent, the helper's choice overriding the circuit.
- **GPU controls on the fixtures:** the free runs end EOS with no lock error, and every token sent equals the
  history's text inputs. A teacher-forced oracle isn't needed: with token input, the rows the inside-check compared
  don't exist, and the computer side is unchanged from stage 2's oracle.

## The registration (as stage 3, except where noted)
- **Questions:** inside-4b's 160 (`e3f98862...`), in order; one run of F, then one of G, in the same V100 session
  order F then G; no reruns or substitutions.
- **Budgets:** 8,192 text and 2,000,000 chip tokens per question, the reference's separate cap of 2,000,000 records
  per call, helper context 16,384; greedy, the lowest id at a tie.
- **Preflight:** per arm, its GGUF and helper must pass the stage-0 gate (token mode) in the same session.
- **Primary comparison: F minus C**, as stage 3 (Newcombe method 10, 95%, ±5-point margins, inconclusive unless
  the interval says otherwise; the latency rule median F <= 2 x median C and <= 60 s).
- **Secondary, reported regardless:** G minus C, F minus S, G minus S (the same method, labelled secondary; no
  multiplicity correction claimed); the speed ratios (median and summed) of F and G vs S and C.
- **Validity:** any lock error invalidates that arm (its run stops); an exception or OOM stops it as incomplete; one
  disclosed restart from the beginning per arm for infrastructure interruption. Memory by stage 3's method; no
  comparison with C's PyTorch figure.
- **Framing:** F and G are "Qwenputer's exact computer with the stock Qwen language model". By the inside-check
  that's functionally Qwenputer up to the norm rounding, but it's a different set of weights, so it's reported as
  such: not "Qwenputer in llama.cpp".

## Stop rules
- A preflight gate failure for an arm: that arm doesn't run; report.
- A lock error: the arm is invalid; report; no fix-and-rerun inside this registration.
