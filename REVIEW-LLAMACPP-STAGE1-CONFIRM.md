# Quick confirm for Codex: stage 1 findings 1-3 (after 50b81e2 / 000ee0d)

Your own clone, your branch `review/codex-llamacpp-stage1`: merge `llamacpp-port` (this commit). Review only: CPU
only (no GPU job), nothing pushed to `main` or `llamacpp-port`, no edits under `/mnt/nvme/flashnext-test/`, scratch
in your own /tmp folder. The rebuilt helper binary is `/mnt/nvme/flashnext-test/llamacpp/qwen_half_helper`; the one
that ran stage 1 is kept as `qwen_half_helper.v1` (sha256 9383c566...).

1. **Blind first:** re-run your own counterexamples from findings 1 and 2 against the new `helper_client.py`,
   `qwen_half_helper.cpp` and `test_run_plain_llama.py` (the BytesIO NaN frame, invalid ids, the production backend's
   positions with a deliberate offset). Try one mutation of your own on the new controls to see they can fail.
2. Then read the diff `2fa7d34..e58a254` (code, `results/llamacpp-stage1.md`, the `LLAMACPP-PLAN.md` additions for
   stage 2 and the stage-3 memory method), and say whether anything is wrong, missing or still overstated.

Append "Confirm" to `REVIEW-LLAMACPP-STAGE1-FINDINGS.md`, commit, push, and say whether finding 1 is closed.
