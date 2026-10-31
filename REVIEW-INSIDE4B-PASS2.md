# Pass 2 questions: inside-4b and the viewer plan

Only after pass 1 is pushed. Now read the commit messages on `inside-4b`.

1. Where does this account disagree with your pass 1?
2. **Prediction failure:** before the plain-B pilot I predicted that Qwenputer and plain 4B + Python would write
   near-identical replies (same Qwen weights, the same response tokens written in). They matched on 1 of 15; replies
   split within the first sentence even with no tool call. My explanation: the widened residual (3,968 vs 2,560) and
   the bf16-rounded norm rescale change the arithmetic enough to flip early greedy choices. Supported? Other causes
   (Qwenputer's fast path vs plain HF modules, the chip lanes' values in the residual, the halt-echo newline)?
3. **Budgets:** C's `max_chip` was raised from upstream's 60,000 to 2,000,000 after the pilot (it cost C its one
   pilot miss); R gets `MAX_RECORDS` 2,000,000. Equivalent? Note C counts chip *tokens*, R counts *records*: a record
   is several tokens, so the caps aren't the same unit. Which way does that tilt, and does it matter at these sizes?
4. **The full-run scope:** only C and R on the 160 primary questions; plain alone and plain + Python only in the
   pilot. Fair to report the full run's verdict as "inside vs beside" on that basis?
5. **Scorer changes on this branch:** C vs R primary pairing (`other = "B" if "B" in arms else "R"`); arm R accepted.
   The key `C_minus_B` is now used for C minus R. Any risk to stage 5's frozen scoring on `main`?
6. **Viewer:** which phase first, and what would make phase 3's "live view of the computer inside the model" honest
   enough to show other people?
