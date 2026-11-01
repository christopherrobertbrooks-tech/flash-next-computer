# Stage 2 review, pass 2 (open only after pass 1 is pushed)

Now read `results/llamacpp-stage2.md` and compare it with your pass 1. Questions:
1. Is anything on the page overstated or missing (the pilot speed "43.5 vs ~30", "runs at llama.cpp's own speed",
   the ~14 GB-per-token explanation, the pipe-cost figure)?
2. **The stage-3 configuration.** I propose registering the **bf16 output head** GGUF as stage 3's primary
   configuration instead of the f32 head the plan names: its stored values are bit-identical to the source, it picks
   the same argmax at all 14,745 fixture positions, passes the stage-0 gate, is 10.6% faster and 1.9 GB smaller. Is
   that a sound pre-registration (made before any stage-3 data), or should stage 3 stay on the f32 head, or run both?
3. **The stage-3 protocol.** What exactly should be frozen before the run: the 160 questions, budgets, the GGUF and
   helper hashes, the runner commit, the comparison (Newcombe, ±5 margin, median rule vs inside-4b's C), the memory
   method (and whether to re-measure C the same way for a like-for-like memory claim), the stop/abort rules?
4. Anything in the runner that must change before a 160-question run (e.g. the helper's 16,384 context against the
   8,192-text budget plus prompts; the lock's records allowance; what happens on a lock error mid-run)?
Append "Pass 2" to `REVIEW-LLAMACPP-STAGE2-FINDINGS.md`, commit, push.
