# Quick confirm for Codex: early-stop findings 1-5 (after 8caae02 / 068c6da)

Your own clone, your branch `review/codex-earlystop`: merge `earlystop` (this commit). CPU only; nothing pushed to
`main` or `earlystop`; read-only under `/mnt/nvme/flashnext-test/`; scratch in your own /tmp. Don't run
`earlystop.sh`.

1. **Blind first:** re-run your own counterexamples against the new code: the synthetic matching live run with an
   incomplete startup, wrong arm/mode, a zero/other runtime manifest and lock/harness errors
   (`llamacpp/earlystop_live_check.py`); the "keep" before any live run (`earlystop_eval.py`, `earlystop_analysis.py`);
   the -100 s decision times, NaN, reversed times, inconsistent indices, and a negative live time. Try a mutation of
   your own on the new controls (`llamacpp/test_earlystop.py`).
2. Then read the diff `03c2006..` (this commit): the monotonic clock and `expect_sha256` in `run_split.py`, the
   rebuilt runtime manifest and re-frozen analysis identity (the first versions kept as `*.v1-superseded.json`), and
   the development page's two wording fixes. The rule (`c29310c9...`) and the fresh questions are unchanged.
3. Say whether findings 1-5 are closed and whether `earlystop.sh unchanged` may run.
Append "Confirm" to `REVIEW-EARLYSTOP-FINDINGS.md`, commit, push.
