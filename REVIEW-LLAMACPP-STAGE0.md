# Review brief for Codex: llama.cpp port, stage 0 (premise)

Your own clone, a new branch `review/codex-llamacpp-stage0` from `llamacpp-port` (531f971). Review only: nothing
pushed to `main` or `llamacpp-port`; no edits to `/mnt/nvme/flashnext-test/llamacpp/` (copy what you need to your own
/tmp folder); no installs. CPU checks as you like. **At most two short V100 jobs**, under the lock
`/mnt/nvme/humming-fast/tools/v100`, pinned to the V100's UUID; never the 4070. Confirm the V100 is released.

## Pass 1 (blind): push before opening the pass-2 file
Read only `LLAMACPP-PLAN.md` (the plan and its fixed stop rule), the code in `llamacpp/`, and the raw data in
`results/llamacpp/stage0/` plus the gateway files (fixtures in `fixtures-v1/`, `qwen-half/`, the GGUFs, the logs).
Don't read `results/llamacpp-stage0.md`. From that alone:
1. Did stage 0 do what the plan says (0a-0d)? Do the fixtures record what the trusted path actually computed (hooks,
   ordering, ownership, positions)? Could the capture have changed what ran?
2. Is the cut faithful (renaming, the slice, the config, the tokenizer), and the GGUF (types, layer pattern, rope,
   norms, the f32 embedding)?
3. Does the helper feed llama.cpp what the fixtures hold (rows, positions, sections, reset), and do the scorer's
   positions and denominator match the plan? Could any check pass when it should fail?
4. Recompute the stop rule's numbers yourself, and state your own verdict.
Write `REVIEW-LLAMACPP-STAGE0-FINDINGS.md` ("Pass 1"), commit, push.

## Pass 2 (guided)
Then open `REVIEW-LLAMACPP-STAGE0-PASS2.md` and follow it.
