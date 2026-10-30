# Pass 2 questions: stage 5 results

Only after pass 1 is pushed. Now read `results/stage5-results.md`.

1. Where does the write-up disagree with your pass 1? Who's right?
2. **B vs C divergence.** B repeated itself 40/40, yet B and C split before any tool result on 21 questions (first
   tool call different; reasoning diverging within 8-1,440 characters), with identical prompt token counts, one
   identical prefill chunk, nothing reused and identical startup free memory. The write-up says "the computer's
   presence on the card changes Flash-Next's output somewhere in Strata's generation" and suggests memory pressure
   during decoding. Is that supported? What else could explain it (Strata's elastic K/V, slot refill after prefill,
   CUDA context switching, suffix drafts, the tool list or server info rendered into the prompt)? What evidence in the
   engine and server logs would distinguish them? A diagnostic repeat of C on the 40-question subset was then run
   (`results/stage5-final/diag-Crep/`): it matched Python 40/40, and the first C run's differences begin at request 122
   and persist to the end, tool-free questions included. What does that point to?
3. **The out-of-memory event** on `final-5b-007` (4 computer calls, 28 MiB free): fairly reported? Does it bear on
   the divergence?
4. **The verdict:** "practically equivalent" with +0.6 points [-3.0, +4.4] and 5 discordant pairs, all chains.
   Sound under the pre-registration, given the divergence above adds path noise to the pairs?
5. **"Recovered" counts** (B 37, C 43; mostly over-capture refusals then a split): any of them hiding a failure the
   protocol says should count?
6. Does the final section answer Chris's finish line fairly, without overclaiming either way?
