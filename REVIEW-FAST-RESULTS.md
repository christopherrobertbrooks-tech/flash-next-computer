# Review brief for Codex: the registered fast runs (arms F and G)

Your own clone, your branch `review/codex-fast`: merge `inside-check` (this commit). Review only: nothing pushed to
`main` or `inside-check`; read-only under `/mnt/nvme/flashnext-test/` (the runs are in `llamacpp/fast-run/`);
scratch in your own /tmp. CPU checks as you like. **At most two short V100 jobs** if you need them (under the lock,
the V100's UUID); never the 4070.

## Pass 1 (blind): push before opening the pass-2 file
Read only `FAST-PLAN.md` (the registration), the frozen identities (`results/fast/fast-artifacts-*.json`,
`results/fast/fast-analysis.json`), the tools (`llamacpp/fast.sh`, `llamacpp/fast_analysis.py`, `llamacpp/run_split.py`,
`llamacpp/split_chat.py`) and the raw data: `results/fast/F/`, `results/fast/G/` (replies, startup, memory samples,
preflight, arm.out), `results/fast/scores/`, `results/fast/run.out`, the gateway's `fast-run/`, and the saved
comparators (`results/inside4b-full/full-C`, `results/llamacpp/stage3`, and stage 1's `results/llamacpp/stage1-full`).
Don't read `results/fast-results.md` or the commit messages after 96c22f9.
1. Did the runs follow the registration (identity checks, both preflights, order, budgets, validity, no reruns)? Do
   the gateway's files still match the frozen identities?
2. Recompute all four comparisons from the replies yourself (pairs, differences, intervals, verdicts, latency
   rules), the speed figures (medians, sums, buckets), stops and budget-stop time, lock errors and checked chip
   tokens, and the memory figures. Do your answer extraction and the scorer agree?
3. Anything in the data that contradicts validity (an early end, a hidden error, a timing anomaly, a memory gap)?
4. Your own reading of what the fast runs show and don't.
Write "Results pass 1" into `REVIEW-FAST-FINDINGS.md`, commit, push.

## Pass 2 (guided)
Then open `REVIEW-FAST-RESULTS-PASS2.md` and follow it.
