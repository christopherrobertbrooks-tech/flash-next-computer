# Inside-4B: a computer inside the weights vs the same computer beside the model (2026-10-30)

## The question
With the same 4B language model, the same RISC-V CPU and the same calculator program, does it matter whether the CPU
lives **inside the model's weights** (Qwenputer-4B) or runs **beside the model** as a small Python emulator?

## Arms (full run)
| Arm | Model | Computer |
|---|---|---|
| **C** inside | Qwenputer-4B (`divagr1925/Qwen3.5-4B-Qwenputer`, local `/mnt/nvme/qwenputer/model`) | the RV32 circuit compiled into its weights; upstream's executor path, every computer token checked against upstream's reference machine |
| **R** beside | plain Qwen3.5-4B (`Qwen/Qwen3.5-4B`, Apache 2.0, local `/mnt/nvme/qwen35-4b/model`) | upstream's Python RV32 reference machine running the same calc firmware (`calc-bridge.elf`); its response tokens are exactly the circuit's |

The same Hugging Face runtime and decoding for both: greedy, thinking off (Qwenputer's own chat setting), the same
chat template and prompt ids, one token per step, at most 8,192 text tokens per question (tool responses included,
as upstream counts), at most 2,000,000 computer steps per question (C: `max_chip`; R: `MAX_RECORDS`).
Qwenputer's Qwen weights are the base model's, unchanged apart from a norm rescale for its wider layers (verified
tensor by tensor). Scripts: `inside4b/run_qwenputer.py`, `inside4b/run_plain.py --arm R`.

Not in the full run (pilot only, `results/inside4b-pilot/`): plain alone (2/12) and plain + ordinary Python (11/12;
replies identical to R on 14 of 15).

## Questions
The 160 primary questions (groups 1-4) of stage 5's frozen final set: `inside4b/final-primary.jsonl`, sha256
`e3f98862352ad0111c5b5428a3580d01ebc0fcec3474b8a495300e577ca7074d` (the groups 1-4 rows of
`stage5/final-questions.jsonl`, unchanged). Scored by `stage5/score.py`.

## Decision rule (as in stage 5)
C minus R in accuracy, paired, Newcombe 95% interval; practically equivalent if inside +/-5 points; better/worse if
entirely above/below 0; otherwise inconclusive. "Practical": C's median seconds per question at most 2x R's and at
most 60 s. A computer-vs-reference mismatch in C is a hard failure.

## Prediction (written before the run)
Practically equivalent in accuracy; C about 3x slower (pilot 51 vs 17 s median) with ~2.5x the V100 memory (~20 vs
7.8 GB), so **not practical** by the rule; zero mismatches in C. If so, the computer's safety and checkability come
from the tiny CPU and its firmware, not from being in the weights.

## Order and cost
R first (~1.75 h), then C (~3.5 h), one job at a time in the V100 lock, overnight. Frozen at tag `inside4b-freeze`.
