# Stage 3 review, pass 2 (open only after pass 1 is pushed)

Now read `results/llamacpp-stage3.md` and compare it with your pass 1. Questions:
1. Is anything on the page overstated or missing? In particular the speed lines (1.68x median, 1.48x summed, the
   text bucket 48.0 vs 28.4/s), "the computer stayed exact", and the group-4 remark (18 vs 24, not a registered test).
2. Is it fair to describe the result as "the port works: exact computer, ~1.7x faster per question, accuracy
   inconclusive"? What wording would you use for a one-paragraph public summary (an upstream Qwenputer issue,
   which Chris approves word for word before anything is posted)?
3. What should the final write-up (stage 4) include or caveat: reproducibility (one run each), the bf16-head choice
   on development data, the f16 Qwen layers vs the original bf16, the upstream loading patch, what isn't a native
   llama.cpp port?
4. Anything worth one more measurement before the write-up (e.g. a repeat of a few questions for timing
   variability), or is the evidence complete for what it claims?
Append "Pass 2" to `REVIEW-LLAMACPP-STAGE3-FINDINGS.md`, commit, push.
