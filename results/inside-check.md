# How does Qwenputer's computer reach its language model? (2026-10-31, after the llama.cpp port)

**Answer:** on the executor decoding path, the language model gets the computer's work as text. On the 12 saved
executor histories (where the mute marker `mu` is 0), zeroing every computer input dim preserved all 13,746
Qwen-owned greedy choices. The language matrices keep their base blocks, with added muting controls and norm
compensation, and plain HF Qwen agrees on 99.869% of those choices. This supports testing a stock-model replacement.
It doesn't establish mathematical or free-running equivalence: norm epsilon, stored-gain rounding, engine arithmetic
and quantization can all change outputs. The design is upstream's (docs/q35.md: the exit "clears every chip dim
except the mute marker", and the mute keeps "Qwen's state ... unchanged and pure text ... bit-identical"); these are
our measurements of it on real runs. Reviewed by Codex (review/codex-fast baf76c5 / 739fda6): every number
reproduces; this page carries its corrections.

## What the weights show (CPU, the cut Qwen half; `llamacpp/inside_check_weights.py`, `results/inside-check/checks.txt`)
- **Writes:** none of the 32 Qwen layers writes to any of the 1,408 computer dims; every residual-writing
  projection's rows past 2,560 are exactly 0.
- **Reads:** they read exactly **one** computer dim, `mu` (2776). It feeds the gates `a`, `b`, `z` of all 24
  linear-attention layers and `q`/`k` of the 8 full-attention layers. Upstream's mute sets it to 1 on chip tokens,
  so that a forward over chip tokens leaves Qwen's state unchanged. In the chat loop chip tokens skip Qwen's layers,
  and `mu` was 0 in all 20,486 rows Qwen received on the fixtures.
- **Base weights:** the language matrices' original blocks equal the base checkpoint's (Codex: 360 tensors
  bit-identical), and the 65 norms equal the expected bf16-stored rescaling `(1+w)*sqrt(2560/3968)-1`. The input
  norms' computer extension is stored as w = 0 (a gain of 1).
- **The output head does use computer features:** the final norm keeps gains on the 21 code dims (2^20) and `tau`
  (2^60) and zeroes most others (stored w = -1). The code columns carry the circuit's own output, and `tau` is a
  common offset on text logits. At circuit-owned positions these decide the token.
- **Not an exact identity with plain Qwen:** RMSNorm's epsilon (1e-6 in both) isn't rescaled, so even with ideal
  gains the widened norm computes `1/sqrt(M + eps*3968/2560)` instead of `1/sqrt(M + eps)`. On top of that come the
  rounding of the stored gains and kernel arithmetic. With typical residuals the epsilon effect is tiny, but it means
  "plain Qwen up to rounding" is an approximation.

## What the runs show
| Check | Result |
|---|---|
| llama.cpp, the stage-0 fixture rows as recorded vs with every dim past 2,560 zeroed (prompt and history), teacher-forced (`llamacpp/ablate_chip_dims.py`) | **Qwen-owned positions: 13,746 / 13,746** have the same argmax and the same top-5 ids; logit gaps differ by at most 7.6e-6. **Circuit-owned positions: 465 / 999** have the same argmax; the rest change, as designed, because their output is the circuit's code. (All 14,745 positions: 14,211 the same.) |
| Qwenputer's HF choice vs **plain Qwen3.5-4B** (HF, bf16) on the same text (`llamacpp/agreement_vs_plain.py`) | **13,728 / 13,746 = 99.869%**; 18 disagreements, all near-ties (largest margin 0.25, 12 exact ties). That's high agreement, and also direct evidence the choices aren't bit-for-bit equal; it doesn't identify the cause of each mismatch or cover unseen histories. |

## What it means
- **The interface on this path:** the language model consumes text, and the computer controls its own trace and
  output. That's a statement about the executor decoding path, not about every full-checkpoint path or every
  internal state. So keep three things separate: the compiled weights (the computer is in the checkpoint), the
  executor realization (upstream's fp32 executor computes the circuit), and the language interface (text).
- **inside-4b's 135 vs 130 (inconclusive)** is consistent with this interface. It doesn't prove a mechanism, and it
  doesn't prove the two systems equal.
- **A stock-model replacement** (the stock Qwen3.5-4B next to the executor) is a sound new system to test, framed as
  "Qwenputer's exact computer with the stock Qwen language model": `FAST-PLAN.md`.
