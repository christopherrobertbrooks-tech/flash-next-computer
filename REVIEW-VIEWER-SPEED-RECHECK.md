# Re-check brief for Codex: viewer and speed (after 809ed9c / 2970e32)

Your own clone, your branch `review/codex-viewer-speed`: merge `viewer-live` (ba358d5) and `computer-speed`
(c9114e7) into it. Review only: nothing pushed to `main`, `viewer-live` or `computer-speed`; no installs, no edits
to the gateway's live copies (`/mnt/nvme/flashnext-test/...`); write scratch under your own /tmp folder.
CPU checks as you like. **At most one V100 job** (a headless live viewer run, any question you choose), under the
lock `/mnt/nvme/humming-fast/tools/v100`, pinned to the V100's UUID; never the 4070. Confirm the V100 is released
afterwards.

## Pass 1 (blind): push before opening the pass-2 file
Without reading my commit messages, `VIEWER-PLAN.md`'s status section or `results/speed-results.md`:
1. Re-run your own counterexamples from findings 1, 2, 3 and 9 against the new `viewer/rvlive.py` and
   `viewer/rvview.py`: the display before, during and after a partial record and after a wrong token; every EOF
   point and a worker error; a full queue with a stop; the labels at 80 and 120 columns.
2. Read `viewer/test_rvlive.py` and say whether its controls can fail (try a mutation or two).
3. Anything new you find in the changed code.
Append "Re-check pass 1" to `REVIEW-VIEWER-SPEED-FINDINGS.md`, commit, push.

## Pass 2 (guided)
Then open `REVIEW-VIEWER-SPEED-RECHECK-PASS2.md` and follow it.
