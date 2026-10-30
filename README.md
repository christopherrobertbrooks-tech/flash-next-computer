# flash-next-computer

Can an exact, deterministic computer (from [Qwenputer](https://github.com/divagr18/qwenputer)) work with
Qwen3.8-Flash-Next Coder served by Strata on Chris's Tesla V100, and does it beat the same model alone and the same
model with an ordinary Python tool?

**Finish line:** Flash-Next Coder on the V100 able to call an exact computer, with a measured answer to whether it does
better than Flash-Next alone and Flash-Next with host Python (accuracy, latency, memory).

The working plan is `PLAN.md` (v2, 2026-10-30, as Chris supplied it). The baseline it builds on is the private
`qwenputer-v100` project (Qwenputer runs exactly on this V100 in bf16 on both computer paths; 195/200 x2).

## Status
- 2026-10-30: **Stage 5 done: the measured answer.** 160 maths questions, frozen protocol (`STAGE5-PLAN.md` v3, tag
  `stage5-freeze`): Flash-Next alone 81, with Python 155, with the computer 156. Computer vs Python: **practically
  equivalent** (+0.6 points, 95% interval -3.0 to +4.4), 1.6x slower (practical by the agreed rule), no reference
  mismatches, one question with four out-of-memory call failures. Open: the original long computer run diverged from
  Python on later questions, even before any tool result; a fresh short computer run matched Python on its
  40-question subset. The cause, and any dependence on shared-GPU resource history, remain unresolved. Codex
  reproduced the scores (ff80127 / 3d12646). `results/stage5-results.md`.
- 2026-10-30: Codex review of 60a4a33 (c2d2b77): the stage 4 answer check is now exact (a declared ANSWER line,
  parsed and compared as a value; 25-case self-test), declared backend differences are exact triples, and the input
  inventory covers Flash-Next's tokenizer and chat template. Live re-run with the new prompts: computer and Python
  both pass, identical replies. `results/stage4-bridge.md`.
- 2026-10-30: Stage 4 robustness: two conversations in a row and MTP on/off give identical outputs; vision + computer
  ran out of memory as configured (Strata leaves 486 MiB) and works with `--vram-reserve-mib 1200` (-255 expert
  slots, ~2%); `--prompt-cache 0` turned out not to be a clean pause/resume test (it changes fresh reads too), so
  direct state isolation stays open; no cancellation. Both backends now share one input contract: 121/126 inputs
  identical, the 5 differences are the computer's declared limits (too complex, 62-digit power steps, tab, `1_000`).
  Frozen inputs in `results/inputs-manifest.txt`. `results/stage4-bridge.md`.
- 2026-10-30: Stage 4 first proof: Flash-Next called the computer by itself through Strata's MCP tools, 4/4 exact;
  on these four prompts, the final replies were identical with the computer and with Python (Python twice: identical).
  State isolation isn't shown by that and stays open. Cost: 1.4-3.2 s per call (Python milliseconds or less) and
  ~430 MiB of VRAM (peak 32,417 of 32,768 MiB). Codex's re-check (3443e87): the request driver now enforces declared
  expectations (self-test proves wrong answers fail) and saves every raw reply. `results/stage4-bridge.md`.
- 2026-10-30: Stage 1 done: Codex verified the last 40 paired records locally, no findings (`qwenputer-v100`, `REVIEW-40-FINDINGS.md`): all 200 pairs identical across both computer paths.
- 2026-10-30: Codex reviewed stages 2-3 (`REVIEW-FINDINGS.md`): fail-open paths in the first service and runner,
  memory measured on a stand-in, some overstated wording. The recorded service failures and the runner were improved;
  the 103-case corpus passed and a deliberately wrong expectation failed; the real service uses ~450 MiB GPU and
  ~1.7 GB host RAM. Codex's re-check of `a7ac5b3` (352b583) found two redlines left (exceptions and failed resets left
  the session valid; the time limit skipped the reference run) and two nits (benchmark scripts exited 0 on failures;
  wording). Those are fixed and the gate now has positive controls for each: `results/stage3-gate.md`.
- 2026-10-30: Stage 3 first interface proof (superseded by the gate above): 88/88 clean calls, ~1.6 s per call
  (Python: microseconds). `results/stage3-interface.md`.
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
