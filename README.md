# flash-next-computer

Can an exact, deterministic computer (from [Qwenputer](https://github.com/divagr18/qwenputer)) work with
Qwen3.8-Flash-Next Coder served by Strata on Chris's Tesla V100, and does it beat the same model alone and the same
model with an ordinary Python tool?

**Finish line:** Flash-Next Coder on the V100 able to call an exact computer, with a measured answer to whether it does
better than Flash-Next alone and Flash-Next with host Python (accuracy, latency, memory).

The working plan is `PLAN.md` (v2, 2026-10-30, as Chris supplied it). The baseline it builds on is the private
`qwenputer-v100` project (Qwenputer runs exactly on this V100 in bf16 on both computer paths; 195/200 x2).

## Status
- 2026-10-30: Stage 1 done: Codex verified the last 40 paired records locally, no findings (`qwenputer-v100`, `REVIEW-40-FINDINGS.md`): all 200 pairs identical across both computer paths.
- 2026-10-30: Stage 3 interface proof: the computer as a service, every token checked against the reference
  machine: 88/88 clean calls (+1 budget stop as designed), 70/70 random calls equal to Python; ~1.6 s per call
  (Python: microseconds). Stage 3 gate met. `results/stage3-interface.md`.
- 2026-10-30: Stage 3 Strata side by side: vision off + the computer process (434 MiB) = 12,152 expert slots and
  10.4 s per request, the same as today's vision-on setup (11,642 slots, 10.5 s); Strata alone with vision off 9.4 s;
  computing at the same time 23.3 s (take turns, don't overlap). `results/stage3-strata.md`.
- 2026-10-30: Stage 3 first measurement: the executor alone is ~21 MiB (68 MiB peak), exact on 4,166 steps, ~2,000
  steps/s; as its own process ~440 MiB (368 MiB CUDA context); CPU ~50 steps/s (too slow). `results/stage3-executor.md`.
- 2026-10-30: Stage 2 read-only map done of the **deployed** Strata (v0.1.41, `fb58e0d`, not the public 0.2.x the
  plan's source notes describe) -> `mapping/`.

## Rules
Private until Chris says otherwise. No model downloads, posts or merges without his OK. Read-only on the live Strata
setup until a test is agreed; tests run as their own launch, never by changing llama-swap's live config. Sprocket's
voice stays on the 4070. Codex reviews are two-pass (blind first).
