# Inside-4B: a computer inside the model vs the same computer beside it (v2, 2026-10-30)

v1 (tag `inside4b-freeze`) was stopped at 81/160 of arm R on Chris's yes after Codex's review (124755d / 5f0972b):
R left out the HALT-echo newline that C writes, and the computer budgets differed in unit and scope. Its partial
data are kept (`/mnt/nvme/flashnext-test/inside4b/stopped-v1/`) and used for nothing. v2 fixes both and is
frozen at tag `inside4b-freeze-2`.

## The question
With the same 4B language model, the same RISC-V CPU and the same calculator program, does it matter whether the CPU
lives **inside the model's weights** (Qwenputer-4B) or runs **beside the model** as a small Python emulator?

## Arms (full run)
| Arm | Model | Computer |
|---|---|---|
| **C** inside | Qwenputer-4B (`divagr1925/Qwen3.5-4B-Qwenputer`, local `/mnt/nvme/qwenputer/model`) | its RV32 circuit, run by **upstream's fast circuit executor** (fp32, from the same circuit description), every computer token checked against upstream's reference machine. This run doesn't execute or verify the checkpoint's bf16 computer layers. |
| **R** beside | plain Qwen3.5-4B (`Qwen/Qwen3.5-4B`, Apache 2.0, local `/mnt/nvme/qwen35-4b/model`) | upstream's Python RV32 reference machine running the same calc firmware (`calc-bridge.elf`); every text token it writes after a call goes into the stream, the HALT-echo newline included, as C's circuit writes it (`inside4b/test_streams.py`). |

The same Hugging Face runtime and decoding for both: greedy, thinking off (Qwenputer's own chat setting), the same
chat template and prompt ids (each row records its prompt-id hash), one token per step, exact bf16 matmul reductions
(set and logged in both).

**Budgets, one unit:** at most 8,192 text tokens per question (tool-response tokens included, EOS not counted, as
upstream counts), and at most **2,000,000 computer trace tokens per question, cumulative over all its calls**, in both
arms (C: upstream's `max_chip`; R: the same count of the reference machine's non-text tokens). The reference checks
get record allowances equal to that budget, so neither binds first (C's lock: 2,000,000 records per call instead of
upstream's hidden default of 1,000,000; R: what's left of the budget). A record is one or more trace tokens (e.g.
1,360 records = 2,146 trace tokens for 123456789*987654321), so a record allowance never binds before the token budget.

**Final answers** are read with `inside4b/frames.py`: only assistant text outside complete call/response frames; a
call without a response is an open call (a hard budget failure), so an "ANSWER:" inside an unfinished call can't count.

Qwenputer's Qwen weights are the base model's, unchanged apart from the norm rescale its wider layers need:
`inside4b/verify_base.py` (log `results/inside4b-verify-base.log`): all 426 base language-model tensors present; 361
identical in their base block; the 65 that differ are the norms, with (1+w_q)/(1+w_b) means 0.80320-0.80334 against
sqrt(2560/3968) = 0.80322 (individual elements 0.797-1.000, bf16 rounding).

**What still differs between C and R** (so this compares two implementations, not CPU placement alone): C runs the
widened Qwenputer model (3,968-wide residual, rescaled norms, its own fast decode path) and the circuit executor; R
runs the plain base model through ordinary HF modules and the Python reference machine. Wording differences between
their replies can't be attributed to the CPU's location (the pilot's no-tool replies already differ).

Scripts: `inside4b/run_qwenputer.py`, `inside4b/run_plain.py --arm R`; CPU controls `inside4b/test_streams.py`.

Not in the full run (pilot only, `results/inside4b-pilot/`, made with the v1 runners): plain alone (2/12) and plain +
ordinary Python (11/12; replies identical to R on 14 of 15). **Pilot limits:** C used upstream's 60,000-trace-token
cap (its one miss hit it); B and R missed one question each at the 8,192-text-token limit; R and B lacked the final
newline. The pilot's 51 vs 17 s medians are descriptive, not a matched-budget estimate, and 12 paired questions give
an interval of about +/-28 points.

## Questions
The 160 primary questions (groups 1-4) of stage 5's frozen final set: `inside4b/final-primary.jsonl`, sha256
`e3f98862352ad0111c5b5428a3580d01ebc0fcec3474b8a495300e577ca7074d` (the groups 1-4 rows of
`stage5/final-questions.jsonl`, unchanged). Scored by `stage5/score.py`.

## Decision rule (as in stage 5)
C minus R in accuracy, paired, Newcombe 95% interval (`stage5/score.py --compare C,R`); practically equivalent if
inside +/-5 points; better/worse if entirely above/below 0; otherwise inconclusive. "Practical": C's median seconds
per question at most 2x R's and at most 60 s. A computer-vs-reference mismatch in C is a hard failure.

## Prediction (written before the run)
Practically equivalent in accuracy; C about 3x slower (pilot 51 vs 17 s median) and using more V100 memory (logged
as peak allocated in both runs this time), so **not practical** by the rule; zero mismatches in C. If so, both
implementations offer reference-checkable traces at similar accuracy, with C costing more time and memory.

**How the result will be worded** (Codex): "On these 160 primary questions, Qwenputer's checked fast circuit executor
and plain Qwen3.5-4B with the same calculator firmware on the RV32 reference machine had [paired result]. The two
also differ in model width, numerics and runtime path, so this does not isolate CPU placement on its own."

## Order and cost
R first (~1.75 h), then C (~3.5 h), one job at a time in the V100 lock, overnight. Frozen at tag `inside4b-freeze-2`.
After the run, an audit lists every length, budget and lock stop and each answer's trace tokens, so whether any
budget bound is visible (Codex).

## After Codex's re-check (282c777)
- **Verified by Codex:** every element of all 426 base language-model tensor blocks: 361 bit-identical, and all
  166,400 norm elements exactly `bf16((1 + w_base) * sqrt(2560/3968) - 1)`; the chat template is byte-identical; and
  the rendered prompt ids are identical for **all 160** questions. `verify_base.py` now checks the norms element by
  element too.
- **Exact-cap edge (finding 9), disclosed, not fixed in the running v2:** v2's R stops only when its trace count goes
  *over* 2,000,000, while upstream's C loop stops on *reaching* it. They differ only if a call ends exactly on the cap.
  The pilot's largest answer used 63,223 trace tokens. The post-run audit lists every answer's trace tokens and every
  budget stop, so whether this edge was ever reached will be shown, not assumed. Fixed after the freeze (`>=`, with
  an exact-cap and a cap-plus-one control in `test_streams.py`, both passing) for any later run. R also runs a whole
  call before writing its answer, where C interleaves trace and response tokens; R's counters describe the work done.
- **R's call records (finding 10):** v2's R rows have `call_text` but no `expression`, so the scorer's loop flag would
  see three different calls as one repeated. The scorer now falls back to `call_text` (the loop flag is diagnostic
  only; no verdict uses it). Fixed in the runner after the freeze, with the instruction count corrected for parking.
- These post-freeze changes are on the branch; the running v2 uses the code at tag `inside4b-freeze-2`
  (gateway copies unchanged; hashes checked).
