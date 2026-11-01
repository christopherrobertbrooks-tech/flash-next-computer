# Is Qwenputer's computer "inside" its language model? (2026-10-31, after the llama.cpp port)

**Answer:** the language part's choices depend only on the text. In the decoding loop, the computer reaches Qwen's
layers only through the tokens it writes. That's upstream's own design (docs/q35.md: the exit "clears every chip dim
except the mute marker", and the mute keeps "Qwen's state ... unchanged and pure text ... bit-identical"); these are
our measurements of it on real runs. So the computer is inside the checkpoint's weights, but it talks to the language
model the way a computer beside it would: by writing text.

## What the weights show (CPU, the cut Qwen half)
- **Writes:** none of the 32 Qwen layers writes to any of the 1,408 computer dims (every residual-writing
  projection's rows past 2,560 are exactly 0).
- **Reads:** they read exactly **one** computer dim, `mu` (2776): the gates `a`, `b`, `z` of all 24 linear-attention
  layers and `q`/`k` of the 8 full-attention layers. Upstream's mute (`mute.py`) sets it to 1 on chip tokens, so
  that a stock forward over chip tokens leaves Qwen's state unchanged. In the chat loop chip tokens skip Qwen's
  layers entirely, and `mu` was **0 in every row** Qwen received on the 12 fixtures (5,753 prompt rows, 14,733 text
  steps).
- **The output head** reads the 21 code dims (gain 2^20) and `tau` (gain 2^60), which is how the circuit wins the
  argmax where it owns the next token. At Qwen-chosen positions the code is 0 and `tau` = 2^-40, the same offset on
  every text logit.
- The input norms are 0 on the computer dims (stored as w = 0, so a gain of 1), with the language dims rescaled to
  compensate for the wider residual. Mathematically, on text the half equals plain Qwen3.5-4B.

## What the runs show
| Check | Result |
|---|---|
| llama.cpp, the stage-0 fixture rows as recorded vs with **every dim past 2,560 zeroed** (prompt and history), teacher-forced | at all **13,746 Qwen-chosen positions**: the same argmax and the same top-5 ids; logit gaps differ by at most 7.6e-6 (the rounding of the common offset). At the 999 circuit-owned positions the output changes, as designed (those choices are the circuit's) |
| Qwenputer's HF path vs **plain Qwen3.5-4B** (HF, bf16) on the same text | the same greedy choice at **13,728 / 13,746 = 99.869%** of Qwen-chosen positions; all 18 disagreements are near-ties (largest margin 0.25, 12 exact ties) |

## What it means
- **Inside vs beside:** with the same text, Qwenputer's language model decides like plain Qwen, up to the rounding
  of its rescaled norms. That fits inside-4b's result (135 vs 130, inconclusive): the computer in the weights gave
  the language model no channel a computer beside it lacks. What differed there was numerics and runtime, and that
  explanation fits the data but isn't proven for each divergence.
- **What "inside" does buy:** the circuit's own tokens are produced by the model's layers (the token stream, not a
  tool protocol), and they're exact and checkable. On this path upstream's executor computes them from the circuit
  description.
- **For speed:** the language half could be run as the stock 2,560-wide Qwen3.5-4B (no widened layers; 8.1 vs
  14.0 GiB) next to the executor. That would be a different model only by the norm rounding measured above, and
  needs its own check.

Raw: `results/inside-check/ablate-chip.json.gz` (both passes' top-5 at every position); script
`llamacpp/ablate_chip_dims.py`. Not yet reviewed by Codex.
