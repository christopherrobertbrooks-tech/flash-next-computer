# Fast-run review, pass 2 (open only after pass 1 is pushed)

Now read `results/inside-check.md`, `results/speed-options.md` and `FAST-PLAN.md`. Questions:
1. Is anything on the two pages overstated (in particular "the language part's choices depend only on the text",
   "inside vs beside", and the whole-question speed estimate)?
2. **The plan:** is running the stock Qwen (token input) next to Qwenputer's executor a sound test of what it claims,
   given the inside-check? Is the framing right ("Qwenputer's exact computer with the stock Qwen language model", not
   "Qwenputer in llama.cpp")?
3. Is the registration sound: F (Q8_0) primary vs C, G (f16) secondary, F/G vs S descriptive, both arms run, no
   winner-picking afterwards? Should the primary comparator be C or S?
4. What must the build include before the runs (controls, identities, preflight), and is skipping a new
   teacher-forced oracle justified for token input?
Append "Pass 2" to `REVIEW-FAST-FINDINGS.md`, commit, push.
