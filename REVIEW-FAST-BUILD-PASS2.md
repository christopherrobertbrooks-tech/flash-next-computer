# Fast-run build review, pass 2 (open only after build pass 1 is pushed)

Now read `FAST-PLAN.md` (v2 plus its build status). Questions:
1. **The development observation:** on the 12 fixtures, F (Q8_0) hit the text budget on 2 (one with no call) and
   ran longer elsewhere; G (f16) ended EOS on all 12, word-for-word stage 1's plain run. The plan keeps F primary
   (registered before these runs) and runs both. Is keeping the registration as is right, and is the observation
   disclosed properly? Anything about it that should be measured or reported in the run (e.g. a loop indicator)?
2. G's texts are identical to stage 1's L (plain Qwen f16 with the reference emulator beside it). With the same
   model and prompts and an exact computer either way, that's expected. Does it change how F/G vs C should be read
   (for instance, G is close to "stage 1 with the circuit's executor instead of the emulator")?
3. Is anything missing before `fast.sh` may run as registered?
Append "Build pass 2" to `REVIEW-FAST-FINDINGS.md`, commit, push, and say whether the runs may proceed.
