# Review brief for Codex: the early-stop development choice, the freeze and the build

Your own clone, a new branch `review/codex-earlystop` from `earlystop` (this commit). Review only: nothing pushed to
`main` or `earlystop`; read-only under `/mnt/nvme/flashnext-test/` (`earlystop/` holds the development workspace;
`llamacpp/` the deployed build); scratch in your own /tmp. CPU checks as you like. **At most one short V100 job** if
you need one (under the lock, the V100's UUID); never the 4070. **Don't run `earlystop.sh`.**

## Pass 1 (blind): push before opening the pass-2 file
Read only `EARLYSTOP-PLAN.md` up to (not including) its "The frozen rule" section, the code (`llamacpp/earlystop_dev.py`,
`earlystop_rule.py`, `earlystop_eval.py`, `earlystop_live_check.py`, `earlystop_analysis.py`, `earlystop.sh`,
`make_earlystop_artifacts.py`, `test_earlystop.py`, the `split_chat.py` and `run_split.py` diffs since 0f31f26, the
`stage5/generate.py` diff), the raw grid `results/earlystop-dev.json.gz`, the frozen identities in `results/earlystop/`,
and the fresh questions. Don't read `results/earlystop-dev.md` or the commit messages after a8e4a4e.
1. From the grid alone, which rule would you pick under the plan, and is the development gate met? Is the token
   rebuild sound enough for development?
2. Does the build do what the plan specifies: the decision hook's boundary (after the decision, before appending; only
   the model's own choices; EOS never refused), "nothing changes" without it, the recorded times, the offline
   evaluation, the live-check selection and tolerance? Do the controls show it, and can they fail?
3. Freeze order: was the rule frozen before the fresh questions existed (commits, times, the file)? Do the identities
   cover what the runs and the scoring use?
Write `REVIEW-EARLYSTOP-FINDINGS.md` ("Pass 1"), commit, push.

## Pass 2 (guided)
Then open `REVIEW-EARLYSTOP-PASS2.md` and follow it.
