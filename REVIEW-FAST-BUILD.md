# Review brief for Codex: the fast-run build (before any registered run)

Your own clone, your branch `review/codex-fast`: merge `inside-check` (this commit). Review only: nothing pushed to
`main` or `inside-check`; read-only under `/mnt/nvme/flashnext-test/`; scratch in your own /tmp. CPU checks as you
like. **At most two short V100 jobs** (under the lock, the V100's UUID; e.g. a few fixture prompts in token mode);
never the 4070. **Don't run `fast.sh`.**

## Pass 1 (blind): push before opening the pass-2 file
Read only the code and the control output, not `FAST-PLAN.md`'s build-status section or the commit messages after
b8b6deb: the diff `b8b6deb..` of `llamacpp/split_chat.py`, `llamacpp/run_split.py`, `stage5/score.py`, and the new
`llamacpp/fast.sh`, `llamacpp/make_fast_artifacts.py`, `llamacpp/fast_analysis.py`, `llamacpp/test_split_chat.py`,
`llamacpp/test_score_fast.py`, `llamacpp/test_split_tokens_gpu.py`; the frozen identities in `results/fast/`; the
control logs in `results/fast/controls/` (and the gateway's `tokens-gpu*.out`).
1. Does token mode do exactly what your fast-plan review required (ExecRow.start unchanged; the prompt at 0..P-1;
   every processed text input at P+k, echoes and the HALT newline included; no chip token or EOS sent; ownership,
   budgets and lock order unchanged; the helper loses where the circuit owns)?
2. Do the controls show it, and can they fail? Is the GPU check's history reconstruction really independent of the
   loop it checks?
3. Scoring and identities: F and G in the scorer; the four comparisons; the new analysis identity (what it covers,
   what it doesn't); the per-arm runtime identities; the launcher's per-arm isolation (does an F failure really let G
   run, and is every attempt kept?).
Write "Build pass 1" into `REVIEW-FAST-FINDINGS.md`, commit, push.

## Pass 2 (guided)
Then open `REVIEW-FAST-BUILD-PASS2.md` and follow it.
