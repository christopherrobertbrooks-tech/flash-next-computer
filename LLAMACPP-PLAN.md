# Qwenputer's language part in llama.cpp (plan v2, 2026-10-31; for Chris's yes)

v2 carries Codex's plan review (review/codex-llamacpp-port 83904cc): the prompt-prefill path, new fixtures instead of
"historical" rows, a helper/state contract, a pre-registered stop rule, and corrected attribution.

## What you'll have at the end
Qwenputer answering maths questions on the V100 with its **language part running in llama.cpp** and its **computer
still on upstream's exact fp32 executor**, every computer token checked against the reference machine, watchable in
the live viewer, plus a measured answer: **how much faster than today's 16.8 s median per question, at what
accuracy**, on the same 160 questions as inside-4b. Good enough = 0 computer mismatches; the speed is whatever we
measure.

## Why this split (from reading upstream's chat loop)
**During generation after the prompt**, upstream's executor runs every token through the circuit, not the
checkpoint's computer layers. For a text token it builds the input to Qwen's 32 layers (`_exec_hc`) from two parts:
Qwen's own embedding dims and the 23 dims the circuit keeps (21 code dims, tau, mu), with everything else zero. Chip
tokens never reach Qwen's layers. So llama.cpp has to run **Qwen's 32 widened layers fed input rows directly**:
llama.cpp accepts float input rows instead of token ids, and `qwen35.cpp` uses its standard `build_inp_embd`.

**The prompt is different.** Upstream first runs the whole prompt through the full checkpoint, its bf16 computer
layers included. Only then does it snapshot the firmware and start the executor: `ExecRow.start` runs the prompt
from zero circuit state, then `load_init()` keeps the lag history. So the prompt's rows into Qwen's first layer come
from the bf16 computer layers, not from `_exec_hc`. Building prompt rows from the executor is a **candidate** that
stage 0 must prove equal to the HF rows (below), not an assumption.

The circuit stays out of llama.cpp on purpose, because none of llama.cpp's precisions is known to keep it exact:
- fp16 is impossible (computer weights reach 8.5e37);
- fp32 mispredicted 1 of 7,554 teacher-forced positions on the V100;
- bf16 on Volta in llama.cpp is unverified.

**What moves:** the Qwen-layer work being moved lies within the 84.5% text-step bucket. That bucket also holds the
executor step on text tokens, building `_exec_hc` and dispatch, which stay where they are, and the pipe adds work.
The moved share and the new boundary overhead will be measured; the speed gain is unknown.

## Stages (Codex review between stages; I check in with you at each end)
0. **Premise (one or two evenings; CPU plus short V100 jobs).**
   a. **New fixtures, captured on the trusted HF path.** These are new, not a replay of the saved inside-4b answers,
      which have no token ids or circuit values. Upstream's own generate (executor) runs on 12 fixed questions: the
      first 3 by id in each group (final-1-001..003, final-2-001..003, final-3-001..003, final-4-001..003). If none
      of them has two calls, a fixed two-call question is added. Hooks record, per position:
      - prompt and generated ids, and who owns the next token (text or circuit);
      - the text position;
      - the row entering Qwen's first layer: HF prefill rows for the prompt, `_exec_hc` rows after it;
      - HF's top-1 over the text vocabulary and its margin over the second choice;
      - hashes of the firmware, tokenizer, model and code.

      The fixtures are written once, never edited, and both engines are compared on them. They cover the prompt
      (tool syntax included), forced response bytes, the HALT newline and two successive calls. These 12 are also in
      the 160 of stage 3; this is a premise check, not tuning.
   b. **Prompt rows:** build the prompt's rows with the executor-based candidate, and compare them with the captured
      HF prefill rows elementwise, plus the first-token logits. Equal (bf16) passes. If they differ, the candidate
      fails and I report the options: keep the HF prefill, or run the 11 computer layers for the prompt elsewhere.
      No silent substitute.
   c. **Cut and convert:**
      - cut out layers 11-42 (renamed 0-31) with their norm compensation, the final norm, and the tied embedding
        sliced to the 248,320 text rows; the base tokenizer;
      - keep 3,968 input columns and the explicit head dims, layer types and rope settings;
      - convert to an f16 GGUF (~10 GB; 213 GB free) and load it in llama.cpp-mainline.
   d. **Teacher-forced comparison.** Feed the fixture rows into the llama.cpp model and compare the top-1 over the
      text vocabulary with the fixtures, at the same positions with the same mask.
      - **Control:** plain Qwen3.5-4B, HF bf16 vs llama.cpp f16, teacher-forced on the same 12 answers' text tokens.
      - **Hard checks, separate from agreement:** the row sent equals the fixture row bit for bit, positions match,
        and the state resets between questions.
      - **Stop rule, fixed now:** stop if the split's pooled top-1 agreement is more than 1.0 point below the
        control's, or if more than 2 of the 12 questions have a split disagreement where HF's margin is at least 1.0
        logit (not a near-tie).
      - **Denominator:** every position in the 12 fixed questions where Qwen chooses the next token: the prompt's
        last position and each generated position the circuit doesn't own. Positions the circuit owns (chip tokens,
        forced byte echoes, the HALT newline) are excluded. The control uses the same positions, the same text
        tokens before them and the same mask.
      - **The optional two-call question** (added only if none of the 12 has two calls) counts in the hard checks,
        not in the stop rule's agreement or its question count.
      - **Reported:** per-question disagreements, the first divergence and its margin.
      - Passing is a premise check, not proof the free-running answers match. Stages 2-3 still decide.
1. **Matched plain baseline (Codex's first step; one evening).** Plain Qwen3.5-4B f16 in llama.cpp, with the same
   RV32 emulator beside it:
   - the same prompts as token ids, the same tool frames (HALT newline included), the same budgets;
   - 12-question pilot, then all 160;
   - compared with inside-4b's R (130/160, 8.63 s median). It measures the accuracy and latency of the **combined
     engine-plus-precision change** (HF bf16 to llama.cpp f16), not either one alone.
2. **The split runner (one to two evenings).** Upstream's generate loop with Qwen's steps sent to a small llama.cpp
   helper program in this repo. The helper is built against llama.cpp-mainline's library; llama.cpp's source is
   unchanged.
   - **The pipe:** each message is framed with a length and a type: reset, prefill, step, end. The rows are f32,
     3,968 wide, with a finite-value check. The reply is the argmax over the text vocabulary, or an error.
   - **The helper's lifecycle:** a helper error or end of input is passed up as an error, never an answer. The child
     is cleaned up on every exit, and it runs under the parent's V100 lock and UUID, never taking the lock itself.
   - **CPU controls, before any GPU run:**
     - a fresh question resets the attention caches, the recurrent state and the text position, and A, B, A in one
       helper reproduces a fresh A;
     - chip-only spans don't advance Qwen, while forced byte echoes and the HALT newline do;
     - prefill and step-by-step histories agree on the same fixture;
     - a circuit-owned token (text or chip) overrides the helper's choice;
     - a mismatch, budget stop or early end can't count as a completed answer;
     - the prompt and `_exec_hc` rows equal the fixtures bit for bit.
   - The lock checks every chip token, as in inside-4b. The live viewer gets a `--backend llama` switch. 12-question
     pilot.
   - **Added after stage 0/1 reviews (Muse, Codex):** one short diagnostic run with `GGML_SCHED_DEBUG=2` (where each
     op runs, prompt and one-token paths; not timed); the per-token cost of the pipe; and, as a separate variant with
     its own GGUF hash, a **bf16 output head** (the source is bf16, so the stored values are identical; check the
     round trip bitwise, the placement, speed, and the stage-0 gate before using it). The baseline stays the f32 head.
3. **Full run (one evening, ~1 hour).** Pre-registered like inside-4b: 160 questions, the same budgets, Newcombe
   interval with ±5 margin against inside-4b's C (135/160, 16.77 s median, 18.98 GiB), median time, 0 lock errors
   required (any mismatch fails this design).
   - Recorded: the clocks, the timing boundaries, and GPU memory measured the same way for both configurations:
     nvidia-smi samples at a fixed cadence (prefill, generation, shutdown), each cooperating process by PID on the
     V100's UUID, summed per sample, then the maximum of the sums; baseline and after-load values too; a failed query
     is "unavailable", never 0. It's a sampled footprint, not an allocation peak, and the historical 18.98 GiB
     (PyTorch peak allocated) can't be converted into it: re-measure the original C configuration the same way if a
     like-for-like claim is wanted.
   - It's port-versus-original evidence (engine, precision and process split together), not an isolated engine
     effect.
4. **Write-up and your call on sharing:** a private results page; an optional upstream Qwenputer issue (I draft, you
   approve; nothing to llama.cpp, whose policy bans AI-written posts).
5. *(Only if you want it after stage 4)* the circuit natively in llama.cpp: an exact executor op, or a precision
   proven exact.

## What I'm unsure of (stage 0 settles it)
- Whether the executor-built prompt rows equal HF's prefill rows (0b).
- Whether llama.cpp's multi-section rope for qwen35 matches HF's interleaved rope at text-only positions.
- Whether 32 layers at hidden 3,968 trips anything keyed off the layer count (`LLM_TYPE_UNKNOWN`).
- The helper's per-token overhead. If the pipe eats the gain, the fallback is llama-cpp-python, which needs your OK
  for a download and an sm_70 build.

## Rules
One V100 job at a time via the lock, never the 4070. No downloads expected. Work on branch `llamacpp-port`, merge on
your yes. Results stay private until your word.
