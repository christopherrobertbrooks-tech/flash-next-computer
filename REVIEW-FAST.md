# Review brief for Codex: the inside-check, the speed options, and the fast-run plan

Your own clone, a new branch `review/codex-fast` from `inside-check` (this commit). Review only: nothing pushed to
`main` or `inside-check`; read-only under `/mnt/nvme/flashnext-test/`; scratch in your own /tmp. CPU checks as you
like. **At most two short V100 jobs**, under the lock and the V100's UUID; never the 4070. Nothing is built for the
fast run yet; please don't run it.

## Pass 1 (blind): push before opening the pass-2 file
Read only the scripts and raw data, not the pages (`results/inside-check.md`, `results/speed-options.md`) or the
commit messages after 24dc718:
- `llamacpp/inside_check_weights.py`, `llamacpp/ablate_chip_dims.py`, `llamacpp/agreement_vs_plain.py`;
  `results/inside-check/` (`checks.txt`, `ablate-chip.json.gz`); the cut Qwen half on the gateway
  (`/mnt/nvme/flashnext-test/llamacpp/qwen-half/`); upstream's `docs/q35.md`, `chipformer/q35/mute.py` and
  `merge.py` (read-only);
- `results/speed-options/` (the two match jsons, the four benchmark jsons, the quantize logs, the GGUF hashes) and
  the stage-0 files they're scored against.
1. What do the weights and the zeroing test establish about how Qwenputer's language half depends on the computer's
   dims? Is the zeroing test sound (positions, what was zeroed, prompt and history, the comparison), and can it
   fail?
2. Is the "the language half equals plain Qwen up to norm rounding" reading supported (the weights, the norm
   rescaling, the 99.869% agreement)?
3. Recompute the speed-options table (agreements via the stage-0 gate, benchmarks).
Write `REVIEW-FAST-FINDINGS.md` ("Pass 1"), commit, push.

## Pass 2 (guided)
Then open `REVIEW-FAST-PASS2.md` and follow it.
