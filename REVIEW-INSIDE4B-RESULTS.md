# Review brief for Codex: inside-4b results (two passes)

The frozen v2 run is done. Please review it as you reviewed stage 5's results.

## Where to work
Your own clone, branch `review/codex-inside4b-results` from `inside-4b`. Review only: no GPU, no model loads, nothing
pushed to `main` or `inside-4b`. Gateway, read-only: `/mnt/nvme/flashnext-test/inside4b/full-R`, `full-C` (also
`stopped-v1/`, which must not be used for any result). CPU reads of checkpoint tensors and tokenizer files are allowed,
read-only, one tensor at a time.

## Pass 1: blind (commit and push before opening `REVIEW-INSIDE4B-RESULTS-PASS2.md`)
**Read:** `INSIDE4B-PLAN.md` (v2 and its re-check section), the code at tag `inside4b-freeze-2`, the raw data in
`results/inside4b-full/` (gzipped replies, run.out, startup.json, questions.sha256, run-full-v2.log), and
`stage5/score.py` (the post-freeze diff since the tag: the `call_text` fallback for R rows).
**Don't read yet:** `results/inside4b-results.md`, `README.md`'s status, commit messages after the freeze tag, or
`REVIEW-INSIDE4B-RESULTS-PASS2.md`.

Do:
1. Check the run matched the frozen protocol (hashes of code and questions, settings, order, budgets, nothing rerun).
2. Rescore independently from the raw rows: right answers per group and arm, the C-vs-R paired table, the Newcombe
   interval and verdict, the practical rule, failure layers.
3. The audit items you asked for: every stop reason, each answer's trace tokens, any budget stop near the 2,000,000
   cap, lock errors, open calls.
4. Anything that would make the numbers or the verdict misleading.
Write `REVIEW-INSIDE4B-RESULTS-FINDINGS.md` ("Pass 1 (blind)"), commit and push.

## Pass 2: guided
Open `REVIEW-INSIDE4B-RESULTS-PASS2.md`, answer it in the same file as "Pass 2 (guided)", push.
