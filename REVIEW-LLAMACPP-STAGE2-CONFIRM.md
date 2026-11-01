# Confirm for Codex: stage 2 findings 1-5 and the stage-3 registration (after dae624e / c23fe76)

Your own clone, your branch `review/codex-llamacpp-stage2`: merge `llamacpp-port` (this commit). Review only:
nothing pushed to `main` or `llamacpp-port`; read-only under `/mnt/nvme/flashnext-test/`; scratch in your own /tmp.
CPU checks as you like. **At most one short V100 job** if you need one (under the lock, the V100's UUID); never the
4070. The stage-3 run hasn't started; please don't start it.

1. **Blind first:** re-run your own counterexamples for findings 1, 2, 4 and 5 against the new code
   (`stage5/score.py`, `llamacpp/run_split.py`, `llamacpp/compare_histories.py`, `llamacpp/test_stage0_gate.py`):
   arm S scoring and an unknown arm; a lock error with right-looking text; repeated PIDs across two GPUs and the
   shutdown phase; an immediate-EOS helper; a corrupted post-divergence row. Try a mutation of your own on the new
   controls.
2. Then read the diff `fb26912..` (this commit): the history oracle (`record_split_history.py`, `oracle_history.py`,
   `compare_histories.py`) and its result (`results/llamacpp/stage2-oracle/`, the 202 MB histories and oracle rows
   are on the gateway in `histories-v1/` and `oracle-v1/`), the stage-3 registration in `LLAMACPP-PLAN.md` v3,
   `llamacpp/stage3.sh`, `llamacpp/make_stage3_artifacts.py` and `results/llamacpp/stage3-artifacts.json` (106
   files; upstream's one local change in `results/llamacpp/upstream-chat-py.diff`), and the corrected stage-2 page.
3. Say whether the findings are closed and whether stage 3 may run as registered.

Append "Confirm" to `REVIEW-LLAMACPP-STAGE2-FINDINGS.md`, commit, push.
