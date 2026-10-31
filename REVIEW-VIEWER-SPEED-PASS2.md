# Pass 2 questions: the viewer and the speed research

Only after pass 1 is pushed. Now read `results/speed-results.md` and the commit messages.

1. Where does the write-up disagree with your pass 1? Who's right?
2. **"The CPU round trip costs only ~5%":** timed2 (~460 us/step) vs graph + a host read per token (~485 us). Does
   the round-trip loop really reproduce today's per-token work (the real decode reads more, and Python decodes the code
   bits)? Is the "~20% Python bookkeeping" estimate (1,712 vs 2,150 tokens/s) sound?
3. **The kernel profile:** 36 kernels, ~500 us; "apply" 21%. Is folding the apply kernels into the producers really
   exactness-preserving, given the FFN kernel's atomic adds into the output buffer?
4. **"Emulator drafts, circuit verifies doesn't pay on this card":** the HF prefill ran at 2.6-2.9k tokens/s on the
   torch fallback and 221/s with fla. Is the conclusion (needs a Volta-specific parallel kernel) right, or is there a
   cheaper route (e.g. Executor2 in fp32 batched over positions for the FFN sublayers, sequential only for the heads)?
5. **The text-speed comparison:** llama.cpp tg128 at 74 tokens/s vs HF decode 44-48 tokens/s (fla probe) vs 37 in the
   runs. Are these comparable (prompt length, cache, greedy argmax in Python vs llama-bench's synthetic decode)? Is
   "a llama.cpp port would cut Qwenputer's total time by a third to a half" a fair estimate?
6. **The fla test:** "fla is slower on Volta because it computes in bf16": supported, or is it compile/autotune time
   in a short run? Was anything about the comparison unfair to fla?
7. **Next project:** is a llama.cpp port of Qwenputer (Codex's route 2) the right next step, given these numbers?
