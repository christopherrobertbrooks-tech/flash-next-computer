# Stage 0 review, pass 2 (open only after pass 1 is pushed)

Now read `results/llamacpp-stage0.md` and compare it with your pass 1. Questions:
1. Is "PASS" right under the plan's rule, and is anything on the page overstated?
2. 0b found the executor-built prompt rows bit-identical to the HF prefill rows. Is the comparison sound (dtype, which
   tensor the hook saw, the executor state before and after), and does it justify building prompt rows that way in
   stage 2?
3. The f16 range finding: the embedding (also the output head) and the final norm have values up to ~1e12 / ~1e18 in
   the computer's columns, so the embedding is kept f32 (3.7 GiB read per output step). Is keeping it f32 right? Is
   there a safe way to check whether those columns contribute to the text logits at all (they'd be droppable only if
   the residual's computer dims after the final norm are exactly zero, or contribute nothing)?
4. The control (plain HF bf16 vs plain llama.cpp f16) and the split were both incremental after one prefill. Is that a
   fair match? Anything about ties (bf16 logits tie often: the largest disagreement margin was 0.125) that changes the
   reading?
5. Before stage 1 (plain Qwen in llama.cpp with the emulator beside it, 160 questions) and stage 2 (the split runner):
   anything in stage 0's tools that must change first?
Append "Pass 2" to `REVIEW-LLAMACPP-STAGE0-FINDINGS.md`, commit, push.
