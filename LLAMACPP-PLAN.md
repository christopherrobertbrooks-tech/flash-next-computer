# Qwenputer's language part in llama.cpp (plan v1, 2026-10-31; for Chris's yes)

## What you'll have at the end
Qwenputer answering maths questions on the V100 with its **language part running in llama.cpp** and its **computer
still on upstream's exact fp32 executor**, every computer token checked against the reference machine, watchable in
the live viewer, plus a measured answer: **how much faster than today's 16.8 s median per question, at what
accuracy**, on the same 160 questions as inside-4b. Good enough = 0 computer mismatches; the speed is whatever we
measure.

## Why this split (from reading upstream's chat loop)
With the executor, upstream already runs every token through the circuit on the executor, not the checkpoint's
computer layers. For a text token it then builds the input to Qwen's 32 layers from two parts: Qwen's own embedding
dims and the 23 dims the circuit keeps (21 code dims, tau, mu), with everything else zero. Chip tokens never reach
Qwen's layers, so Qwen's caches and positions see the text alone.

So llama.cpp only has to run **Qwen's 32 widened layers, fed those input rows directly** (llama.cpp accepts input
embeddings instead of token ids; `qwen35.cpp` uses its standard `build_inp_embd`). Four of the five behaviors Codex
listed then come for free:
- chip tokens skip Qwen;
- the positions are separate;
- the firmware is injected on the executor side, as now;
- the circuit owns the output, as now.

The fifth, reading chip tokens from the code dims, also stays on the executor. Text is 84.5% of today's time, and
this moves exactly that part.

The circuit stays out of llama.cpp on purpose, because none of llama.cpp's precisions is known to keep it exact:
- fp16 is impossible (computer weights reach 8.5e37);
- fp32 mispredicted 1 of 7,554 teacher-forced positions on the V100 (qwenputer-v100);
- bf16 on Volta in llama.cpp is unverified.

A native circuit in llama.cpp is a possible later stage (5), not part of this plan's promise.

## Stages (Codex review between stages; I check in with you at each end)
0. **Premise (one evening; CPU plus one short V100 job).**
   - Cut the Qwen half out of the checkpoint: layers 11-42 renamed 0-31, final norm, and the tied embedding sliced
     to the 248,320 text rows, with the base tokenizer.
   - Convert it to an f16 GGUF (~10 GB on disk; 213 GB free).
   - Load it in llama.cpp-mainline (sm_70 build).
   - Check the 43-to-32 layer config, the layer pattern, the multi-section rope, and whether llama.cpp's input
     embeddings work for qwen35.
   - Teacher-forced test: feed the exact input rows HF would see for 12 saved C answers. Measure top-1 agreement
     with HF's bf16 logits, next to plain-Qwen f16-vs-bf16 agreement measured the same way (the precision control).
   - **Stop rule:** embedding input doesn't work for this model, or agreement is clearly below the plain-model
     control. Then I report and we decide; no workaround hunt.
1. **Matched plain baseline (Codex's first step; one evening).** Plain Qwen3.5-4B f16 in llama.cpp, with the same
   RV32 emulator beside it:
   - the same prompts as token ids, the same tool frames (HALT newline included), the same budgets;
   - 12-question pilot, then all 160;
   - compared with inside-4b's R (130/160, 8.63 s median). This answers on its own whether llama.cpp speeds up the
     "beside" system, and it measures what f16 does to accuracy.
2. **The split runner (one to two evenings).** Upstream's generate loop with Qwen's steps sent to a small llama.cpp
   helper program in this repo. The helper is built against llama.cpp-mainline's library and talks over a pipe:
   input rows in, argmax over the text vocabulary out. llama.cpp's source is unchanged.
   - CPU controls: the input row is bit-identical to upstream's `_exec_hc`; the helper fails loudly on a bad shape.
   - The lock checks every chip token, as in inside-4b.
   - A `--backend llama` switch in the live viewer.
   - 12-question pilot.
3. **Full run (one evening, ~1 hour).** Pre-registered like inside-4b: 160 questions, the same budgets, Newcombe
   interval with ±5 margin against inside-4b's C (135/160, 16.77 s median, 18.98 GiB). Also the median time, peak
   memory, and 0 lock errors required (any mismatch fails this design).
4. **Write-up and your call on sharing:** a private results page; an optional upstream Qwenputer issue (I draft, you
   approve; nothing to llama.cpp, whose policy bans AI-written posts).
5. *(Only if you want it after stage 4)* the circuit natively in llama.cpp: an exact executor op, or a precision
   proven exact. Worth it only if, once text is faster, the computer's share has grown large.

## What I'm unsure of (stage 0 settles it)
- Whether llama.cpp's multi-section rope for qwen35 matches HF's interleaved rope at text-only positions.
- Whether 32 layers at hidden 3968 trips anything that keys off the layer count (Muse flagged `LLM_TYPE_UNKNOWN`
  for 43 layers).
- How far f16 moves the text away from HF bf16. inside-4b already showed small numeric changes alter greedy text, so
  stage 1's control matters.
- The helper's per-token overhead (a pipe round trip per text token). If it eats the gain, the fallback is
  llama-cpp-python, which needs your OK for a download and an sm_70 build.

## Rules
One V100 job at a time via the lock, never the 4070. No downloads expected: both models and llama.cpp are already on
the gateway. Work on branch `llamacpp-port`, merge on your yes. Results stay private until your word.
