# Review brief for Codex: the early-stop runs (unchanged F on the fresh questions, the live check)

Your own clone, your branch `review/codex-earlystop`: merge `earlystop` (this commit). Review only: nothing pushed to
`main` or `earlystop`; read-only under `/mnt/nvme/flashnext-test/` (the runs are `llamacpp/es-unchanged/` and
`llamacpp/es-live/`); scratch in your own /tmp. CPU checks as you like. **At most one short V100 job** if you need
one; never the 4070.

## Pass 1 (blind): push before opening the pass-2 file
Read only `EARLYSTOP-PLAN.md`, the frozen identities (`results/earlystop/earlystop-artifacts.json`,
`earlystop-analysis.json`), the tools (`llamacpp/earlystop*.py`, `earlystop.sh`, `run_split.py`, `split_chat.py`) and
the raw data: `results/earlystop/unchanged/`, `results/earlystop/eval/` (validity, scores, eval, live questions,
final), `results/earlystop/live/`, and the gateway's run folders and logs (`es-unchanged.out`, `es-live.out`). Don't
read `results/earlystop-results.md` or the commit messages after c0305f0.
1. Did the runs follow the registration (identity checks, the preflight, the questions, budgets, validity, the live
   selection procedure, no reruns)? Do the gateway's files still match the frozen identities?
2. Recompute the offline evaluation yourself from the recorded decisions and tokens: the rule's stops, premature
   stops, failures stopped, the estimated saving and medians, the candidate verdict; and the scores. Do they agree?
3. Recompute the live check: token identity, the early-stop flags, times against the tolerance, both runs' validity,
   and the final verdict.
4. Your own reading of what this shows and doesn't.
Write "Results pass 1" into `REVIEW-EARLYSTOP-FINDINGS.md`, commit, push.

## Pass 2 (guided)
Then open `REVIEW-EARLYSTOP-RESULTS-PASS2.md` and follow it.
