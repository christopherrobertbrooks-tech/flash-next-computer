# Review brief for Codex: stage 5 results (two passes)

The frozen stage 5 run is done. Please review the results as you reviewed the tooling.

## Where to work
Your own clone, branch `review/codex-stage5-results` from `stage5`. Review only: no GPU, no Strata, no pushes to
`main`, `stage3` or `stage5`. On the gateway, read-only: the run folders `/mnt/nvme/flashnext-test/stage5/final-*`
(full engine logs `strata.log`, `server.log` and `mem.csv` are only there).

## Pass 1: blind (commit and push before opening `REVIEW-STAGE5-RESULTS-PASS2.md`)
**Read:** `STAGE5-PLAN.md` (v3), the frozen code at tag `stage5-freeze`, `stage5/final-questions.jsonl`, the raw
data in `results/stage5-final/` (replies are gzipped; `calls.jsonl`, `run.out`, configs, startup, memory).
**Don't read yet:** `results/stage5-results.md`, `README.md`'s status, commit messages after the freeze tag, or
`REVIEW-STAGE5-RESULTS-PASS2.md`.

Do:
1. Check the run matched the frozen protocol (the code and question hashes; settings; order; nothing rerun).
2. Rescore independently from the raw replies and call logs (your own code, or `stage5/score.py` plus your own
   spot checks): right answers per group and arm, the C-vs-B paired table, the Newcombe interval and verdict, the
   practical rule, failure layers, group 5, the B repeat.
3. Anything that would make the numbers or the verdict misleading.
Write `REVIEW-STAGE5-RESULTS-FINDINGS.md` ("Pass 1 (blind)"), commit and push.

## Pass 2: guided
Open `REVIEW-STAGE5-RESULTS-PASS2.md`, answer it in the same file as "Pass 2 (guided)", push.
