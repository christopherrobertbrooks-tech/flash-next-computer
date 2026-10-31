# Re-check brief for Codex: inside-4b v2 (after 124755d / 5f0972b)

Your own clone, branch `review/codex-inside4b` (merge `inside-4b` into it). Review only: no GPU, no model loads,
nothing pushed to `main` or `inside-4b`. This time **reading checkpoint tensors on the CPU is allowed**: safetensors
slices or single tensors from `/mnt/nvme/qwenputer/model` and `/mnt/nvme/qwen35-4b/model`, and their tokenizer/template
files, read-only, one tensor at a time (the gateway has 16 GB RAM; the largest tensor is the 3.5 GiB embedding).

What changed (details in `INSIDE4B-PLAN.md` v2 and `VIEWER-PLAN.md` v2):
1. R takes every text token after `</tool_call>` (the HALT newline included); B's frame ends with it too.
   `inside4b/test_streams.py` compares both against the reference machine's own text output.
2. One budget unit for both arms: 2,000,000 trace tokens per question, cumulative; C's lock gets 2,000,000 records
   per call (`functools.partial` on `lock.qwen`); R's reference machine gets what's left of the budget per call.
3. B's `firmware_expression` is a line-by-line port of `gather()`; `test_streams.py` runs your two synthetic cases
   (exactly 80 rows; a NUL) against the real firmware on the reference machine.
4. The executor wording (plan, runner docstring, `startup.json` "path").
5. Pilot limits disclosed in the plan.
6. Scorer: `--compare C,R`; `C_minus_{other}`; chains for R; row-level computer counters. Stage 5 re-scored with the
   modified scorer: all 540 row verdicts, pairs, interval and verdict identical to the published `scores.json.gz`.
7. `inside4b/frames.py`: calls and responses parsed explicitly; open calls flagged (hard failure); the final answer
   only from text outside complete frames. Both runners use it.
8. Viewer plan v2: digit file, record-level steps, verify-before-show, event design, labels, replay data.
Also: `inside4b/verify_base.py` and its log `results/inside4b-verify-base.log` (my base-weights check; please rerun
it or your own).

v1's run was stopped at R 81/160 (data kept, unused). v2 is frozen at tag `inside4b-freeze-2` and may already be
running when you read this; don't look at its outputs. Please re-check 1-8 and append "Re-check" to
`REVIEW-INSIDE4B-FINDINGS.md`.
