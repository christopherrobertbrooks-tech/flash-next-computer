# Quick confirm for Codex: findings 10-11 (after 1eb2056 / 82a4484)

Your own clone, your branch `review/codex-viewer-speed`: merge `viewer-live` (94a587f) and `computer-speed`
(52f8240). Review only: CPU only (no GPU job this time), nothing pushed to `main` or either source branch, no edits
to the gateway's live copies, scratch under your own /tmp folder.

1. **Blind first:** re-run your own counterexamples from findings 10 and 11 and your fake-curses quit test against
   the new `viewer/rvlive.py` (the every-prefix EOF sweep, EOS right after `<tool_call>`, mid-body, real
   `headless()` exit status, a consumer exception in headless, TUI quit before the end). Then try one mutation on
   the new controls in `viewer/test_rvlive.py` to see they can fail.
2. Then read the diff `a59c121..94a587f` and `computer-speed`'s last commit (status line and 84.5%), and say whether
   anything is wrong or still overstated.

Append "Confirm" to `REVIEW-VIEWER-SPEED-FINDINGS.md`, commit, push, and say whether finding 2's redline is closed.
