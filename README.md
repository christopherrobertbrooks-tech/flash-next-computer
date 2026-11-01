# flash-next-computer

Experiments on one Tesla V100 with an **exact computer** for language models: the RV32 computer that
[Qwenputer](https://github.com/divagr18/qwenputer) compiles into Qwen3.5-4B's weights, its fp32 circuit executor, and
its calculator firmware, with every computer token checked against the RV32 reference machine.

## Headline result (2026-11-01)
**Qwenputer's exact computer with the stock Qwen3.5-4B (Q8_0) in llama.cpp answered a registered 160-question maths
set with a 2.33x lower median latency than the original Qwenputer (7.19 vs 16.77 s; 2.40x in summed time).** All
1,447,753 emitted chip trace tokens matched the RV32 reference. Accuracy is still inconclusive: 131/160 vs 135/160,
paired 95% interval -8.3 to +3.2 points. That's one run per configuration, and it's not an exact-weight conversion of
Qwenputer or a native circuit in llama.cpp. Eight answers that ran to the text budget took a quarter of the time.
`results/fast-results.md`.

## What was measured, in order
| Experiment | Question | Result | Page |
|---|---|---|---|
| Stages 1-5 | Can the computer serve Qwen3.8-Flash-Next (Strata) as a tool, vs Python? | 156 vs 155/160 with Python: practically equivalent, 1.6x slower | `results/stage5-results.md` |
| Inside-4B | The computer inside the model (Qwenputer) vs the same firmware beside plain Qwen | 135 vs 130/160, inconclusive; 1.94x the time | `results/inside4b-results.md` |
| Viewer | Watch the computer run, checked token by token | a terminal viewer (`viewer/`), live on the V100 | `VIEWER-PLAN.md` |
| llama.cpp port, stages 0-3 | Qwenputer's language part in llama.cpp, the computer on its exact executor | 131 vs 135/160, inconclusive; 1.68x faster; 0 lock errors | `results/llamacpp-port-results.md` |
| Inside-check | How does the computer reach the language model? | through text: Qwen's layers read only a mute marker, 0 on text | `results/inside-check.md` |
| Fast run | The exact computer with the stock Qwen (Q8_0 primary, f16 secondary) | the headline above | `results/fast-results.md` |
| Early stop | Can runaway answers be stopped early without cutting right ones? (fresh questions) | a frozen repetition rule: 12.2% less time (estimated), 7 of 8 runaways stopped, 0 right answers cut; kept | `results/earlystop-results.md` |
| Fused executor (premise only) | Would fusing the executor's kernels speed the computer up? | no: launch gaps are ~1% of a step; the time is head/FFN kernel work | `results/fused-premise.md` |

The measured comparisons (stage 5, inside-4B, port stage 3, the fast run, the early stop) were registered before their runs, and
every stage was reviewed by Codex in two passes (blind, then guided). Plans: `PLAN.md` (stages 1-5),
`INSIDE4B-PLAN.md`, `VIEWER-PLAN.md`, `LLAMACPP-PLAN.md`, `FAST-PLAN.md`, `EARLYSTOP-PLAN.md`. The first project's log is below.

## Status: closed (2026-11-01)
This line of work is closed. What the exact computer is good for, as measured here:
- **Where it helps:** exact large-number arithmetic for a model that would otherwise guess (Flash-Next alone 81/160,
  with the computer 156/160), with every computer token checked against the RV32 reference.
- **Where it doesn't:** against an ordinary Python tool it's equally accurate and slower, so it doesn't beat it. As a
  general CPU it's far too slow for write-run-fix coding (an 18-line MicroPython snippet takes about 6 minutes on the
  circuit), which is why a planned coding loop was dropped at its premise check.
- **Speed ceilings:** the remaining speed levers are small. Launch fusion is worth ~1%. A device-side loop is worth
  at most ~3% of a question. Faster head/FFN kernels might give up to ~13%, and that would be research.

## History: the first project (stages 1-5)
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
Public since 2026-10-31. No model downloads, posts or merges without Chris's OK. Sprocket's voice stays on the 4070;
experiments use the V100 through its lock. Codex reviews are two-pass (blind first).

## Credits and citation
This project builds on **Qwenputer** by Divyansh Agrawal (<https://github.com/divagr18/qwenputer>, MIT): the
RV32 computer compiled into Qwen3.5-4B's weights, its fp32 circuit executor, the calculator firmware and the reference
machine all come from it. Portions of its code copied here are listed with its license in `THIRD-PARTY-NOTICES.md`.
If you use this work, please cite Qwenputer:

```bibtex
@software{agrawal2026qwenputer,
  author  = {Agrawal, Divyansh},
  title   = {{Qwenputer}: a computer compiled into the weights of a pretrained language model},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/divagr18/qwenputer},
  license = {MIT}
}
```

The language models are Qwen3.5-4B (Apache 2.0); llama.cpp is MIT. Latest results: `results/llamacpp-port-results.md`
(Qwenputer's language part in llama.cpp) and `results/fast-results.md` (the exact computer with the stock Qwen).
