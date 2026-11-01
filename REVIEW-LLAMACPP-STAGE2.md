# Review brief for Codex: llama.cpp port, stage 2 (the split runner)

Your own clone, a new branch `review/codex-llamacpp-stage2` from `llamacpp-port` (this commit). Review only: nothing
pushed to `main` or `llamacpp-port`; no edits under `/mnt/nvme/flashnext-test/` (read-only there; scratch in your own
/tmp folder); no installs. CPU checks as you like. **At most two short V100 jobs**, under the lock
`/mnt/nvme/humming-fast/tools/v100`, pinned to the V100's UUID; never the 4070. Confirm the V100 is released.

## Pass 1 (blind): push before opening the pass-2 file
Read only `LLAMACPP-PLAN.md` (stage 2 and the stage-3 registration), the code (`llamacpp/split_chat.py`,
`llamacpp/run_split.py`, `llamacpp/test_split_chat.py`, `llamacpp/test_split_gpu.py`, `llamacpp/bench_helper.py`,
`llamacpp/qwen_half_helper.cpp`, the `viewer/rvlive.py` backend switch), upstream's `chipformer/rv/chat.py` on the
gateway (read-only), and the raw data: `results/llamacpp/stage2-pilot/`, `results/llamacpp/stage2-bench/` and the
gateway logs (`stage2-pilot.out`, `stage2-bench.out`, `split-gpu.log`, the two `stage2-scheddebug*/` folders).
Don't read `results/llamacpp-stage2.md`.
1. Is `split_generate` upstream's executor-path generate with only the Qwen calls moved (the snapshot and
   ExecRow.start order, ownership, the lock, budgets, EOS, positions, timing buckets)? Is SplitChat's `_exec_hc`
   the same rows without the HF checkpoint?
2. Do the CPU and GPU controls show it, and can they fail? Anything a split-specific failure could slip past
   (a prefix/history mismatch after the first divergence, state left by `exec_prompt_rows`, a helper that
   silently changes)?
3. The diagnostics: placement, the pipe-cost reading, the memory method, and the bf16-head variant's checks.
4. Your own reading of what stage 2 shows and doesn't, and whether stage 3 can be registered on this runner.
Write `REVIEW-LLAMACPP-STAGE2-FINDINGS.md` ("Pass 1"), commit, push.

## Pass 2 (guided)
Then open `REVIEW-LLAMACPP-STAGE2-PASS2.md` and follow it.
