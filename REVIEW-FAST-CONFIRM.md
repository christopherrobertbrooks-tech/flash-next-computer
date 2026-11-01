# Quick confirm for Codex: fast build findings 5-7 (after ef5f988 / bba0138)

Your own clone, your branch `review/codex-fast`: merge `inside-check` (this commit). CPU only (no GPU job); nothing
pushed to `main` or `inside-check`; read-only under `/mnt/nvme/flashnext-test/`; scratch in your own /tmp. Don't run
`fast.sh`.

1. **Blind first:** re-run your own counterexamples against the new code: the final-question lock error with all
   rows present and an "invalid design" startup (`llamacpp/fast_analysis.py`), the max_new=27 cut at `</tool_call>`
   (`llamacpp/test_split_tokens_gpu.py` `reconstruct`), and the launcher's mkdir race (`llamacpp/fast.sh`). Try a
   mutation of your own on the new controls (`llamacpp/test_fast_analysis.py`, `llamacpp/test_reconstruct.py`).
2. Then read the diff `2e36f0a..` (this commit): the rebuilt runtime manifests (`results/fast/fast-artifacts-*.json`,
   only the launcher's hash should differ), the re-frozen analysis identity (`results/fast/fast-analysis.json`; the
   first is kept as `fast-analysis-v1-superseded.json`), and the `FAST-PLAN.md` wording (the G-vs-L counts, the
   tool-call repetition flags, the scoring-validity rule).
3. Say whether findings 5-7 are closed and whether `fast.sh` may run as registered.
Append "Confirm" to `REVIEW-FAST-FINDINGS.md`, commit, push.
