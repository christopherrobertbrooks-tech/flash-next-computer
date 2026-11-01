# Review brief for Codex: llama.cpp port, stage 3 (the registered full run)

Your own clone, a new branch `review/codex-llamacpp-stage3` from `llamacpp-port` (this commit). Review only: nothing
pushed to `main` or `llamacpp-port`; read-only under `/mnt/nvme/flashnext-test/` (the run's own folder is
`llamacpp/stage3-full/`); scratch in your own /tmp. CPU checks as you like. **At most two short V100 jobs** if you
need them (under the lock, the V100's UUID; e.g. re-running a few questions); never the 4070.

## Pass 1 (blind): push before opening the pass-2 file
Read only `LLAMACPP-PLAN.md` v3 (the stage-3 registration), the frozen identities (`results/llamacpp/stage3-artifacts.json`,
`results/llamacpp/stage3-analysis.json`), the launcher and tools (`llamacpp/stage3.sh`, `llamacpp/stage3_analysis.py`,
`llamacpp/run_split.py`, `llamacpp/split_chat.py`), and the raw data: `results/llamacpp/stage3/` (replies, startup,
memory samples, preflight, run.out, scores) and inside-4b's saved C (`results/inside4b-full/full-C`). Don't read
`results/llamacpp-stage3.md` or the commit messages after a335797.
1. Did the run follow the registration exactly (identity checks, the preflight gate, questions and order, budgets,
   the validity rules, no reruns or substitutions)? Do the gateway's files still match the frozen identities?
2. Recompute the result yourself from the replies: the paired table, S minus C and its Newcombe interval, the
   verdict, the latency rule, medians and summed times, lock errors and checked chip tokens, stops, the memory
   figures. Do your answer extraction and the scorer agree on every question?
3. Anything in the data that contradicts the run's validity (an early end, a hidden error, a timing anomaly, a
   memory gap)?
4. Your own reading of what stage 3 shows and doesn't.
Write `REVIEW-LLAMACPP-STAGE3-FINDINGS.md` ("Pass 1"), commit, push.

## Pass 2 (guided)
Then open `REVIEW-LLAMACPP-STAGE3-PASS2.md` and follow it.
