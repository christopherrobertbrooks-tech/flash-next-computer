# Review brief for Codex: llama.cpp port, stage 1 (plain baseline)

Your own clone, a new branch `review/codex-llamacpp-stage1` from `llamacpp-port` (this commit). Review only: nothing
pushed to `main` or `llamacpp-port`; no edits under `/mnt/nvme/flashnext-test/` (copy what you need to your own /tmp
folder); no installs. CPU checks as you like. **At most two short V100 jobs** (e.g. a few questions re-run), under the
lock `/mnt/nvme/humming-fast/tools/v100`, pinned to the V100's UUID; never the 4070. Confirm the V100 is released.

## Pass 1 (blind): push before opening the pass-2 file
Read only `LLAMACPP-PLAN.md` (stage 1), the code (`llamacpp/run_plain_llama.py`, `llamacpp/helper_client.py`,
`llamacpp/qwen_half_helper.cpp`, `llamacpp/test_run_plain_llama.py`, the `stage5/score.py` diff since 531f971), and
the raw data: `results/llamacpp/stage1-pilot/`, `results/llamacpp/stage1-full/` (replies, startup, run.out,
scores) and inside-4b's R run (`results/inside4b-full/full-R`). Don't read `results/llamacpp-stage1.md`.
1. Is arm L really inside-4b's R with only the model calls moved (prompt ids, tool frames, the emulator response and
   the HALT newline, budgets, positions, EOS handling, the timing boundary)? Do the controls show it, and can they
   fail?
2. Does the helper feed llama.cpp correctly in token mode (positions, reset per question, argmax and ties), and does
   the run fail closed on helper errors?
3. Is the scorer change safe (arm L; the C-vs-other code generalized)? Recompute the comparison yourself: pairs,
   interval, verdict, medians, totals, tokens per second.
4. Your own reading of what stage 1 shows and doesn't.
Write `REVIEW-LLAMACPP-STAGE1-FINDINGS.md` ("Pass 1"), commit, push.

## Pass 2 (guided)
Then open `REVIEW-LLAMACPP-STAGE1-PASS2.md` and follow it.
