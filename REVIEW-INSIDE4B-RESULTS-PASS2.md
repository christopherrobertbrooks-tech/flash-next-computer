# Pass 2 questions: inside-4b results

Only after pass 1 is pushed. Now read `results/inside4b-results.md` and the commits after the freeze tag.

1. Where does the write-up disagree with your pass 1? Who's right?
2. **"The computing was identical":** total trace tokens 1,615,594 (C) vs 1,615,658 (R) over 423 vs 419 calls. Is
   "identical computing" a fair summary, or does it need narrowing (different calls on the 72 divergent questions)?
3. **"The 13 disagreements come from the path, not the computer":** replies identical on 88/160 with identical
   verdicts; every disagreement is among the 72 divergent ones. Does that support attributing the 9-vs-4 split to
   numerical path differences in the two copies of the model, rather than to where the CPU lives? What else could it
   be (e.g. C's backend failures: "too complex" on long chained expressions; R's call-format and final-answer
   failures)?
4. **The verdict:** inconclusive, +3.1 points [-1.6, +8.0]. Correct under the pre-registration? Is "leaning to inside,
   by path noise" an overreach in the write-up?
5. **Practical:** C's median 16.8 s vs R's 8.6 s = 1.94x, under the 2x limit. The pilot suggested 3x. Explain the
   difference (question mix? the budget fix?) and whether "practical, just" is fair.
6. **Time breakdown:** 944 s in the circuit executor vs 3.8 s in the emulator; C's text time 6,439 s. Are these
   measured the same way (C's chip_seconds/text_seconds from upstream's generate vs R's per-call ms)?
7. **The scorer's post-freeze fix** (R rows lack `expression`; `computer_error_results` crashed): any effect on
   verdicts? Should the frozen scorer's crash be reported differently?
8. Does the final section answer the question fairly without overclaiming either way?
