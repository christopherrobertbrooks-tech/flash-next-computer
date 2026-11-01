# llama.cpp port, stage 3: the full run (2026-10-31, 5:19-6:55 pm)

**Answer:** Qwenputer with its Qwen half in llama.cpp (bf16-head GGUF) and its computer on upstream's fp32 executor
(arm **S**) scored **131/160** vs inside-4b's original **C** at **135/160**: S minus C = **-2.5 points**, 95%
interval **-8.8 to +3.7** (inconclusive under the ±5 margin: neither a difference nor equivalence is established).
All **1,552,141 emitted chip trace tokens matched the RV32 reference** (0 lock errors); that's
the observed stream on these questions, not a proof over every input, and not a claim that every answer or tool
request was right. The median question took **10.0 s vs
16.8 s (1.68x faster)**; summed time **5,148 vs 7,621 s (1.48x)**; the text-step bucket ran at **48.0 vs 28.4
tokens/s**. This is port-versus-original evidence (engine, precision and process split changed together). Reviewed
by Codex (review/codex-llamacpp-stage3 af2b8ce / a033aba): every number reproduces, no blocking findings; its two
wording corrections are applied below.

Protocol: `LLAMACPP-PLAN.md` v3, registered before the run (runtime identity `results/llamacpp/stage3-artifacts.json`,
106 files, checked at startup: all matched; analysis identity `results/llamacpp/stage3-analysis.json`, verified
before scoring). Preflight in the same V100 session: the stage-0 comparison with this helper and GGUF, gate **PASS**
(99.782% vs control 99.796%, 0 high-margin disagreements). Run status **ok** (no lock error, no exception, 160/160
rows). Scored with `llamacpp/stage3_analysis.py score` (`stage5/score.py --compare S,C`). Raw: `results/llamacpp/stage3/`.

| | S (split, llama.cpp + executor) | C (inside-4b, HF) |
|---|---|---|
| primary | **131/160** | **135/160** |
| group 1 / 2 / 3 / 4 | 69/70, 10/10, 34/40, 18/40 | 69/70, 10/10, 32/40, 24/40 |
| paired | both right 121, S only 10, C only 14, both wrong 15 | |
| failure layers (backend, budget, call format, final answer, reasoning) | 2, 4, 3, 14, 6 | 4, 4, 2, 11, 4 |
| stops | 156 EOS, 4 text budget | 156 EOS, 4 text budget |
| lock errors / chip tokens checked | **0** / 1,552,141 | 0 / 1,615,594 |
| median s per question (outer) | **10.0** | **16.8** |
| summed question time | 5,148 s | 7,621 s |
| text tokens / text-step time / rate | 200,331 / 4,177 s / **48.0/s** | 182,707 / 6,439 s / 28.4/s |
| chip tokens / chip-step time / rate | 1,552,141 / 897 s / 1,730/s | 1,615,594 / 944 s / 1,712/s |
| GPU memory | 17,254 MiB sampled max, both processes (helper 15,540, parent 1,714) | not compared (C's 18.98 GiB is PyTorch's allocated peak) |

- **Accuracy:** inconclusive, as inside-4b and stage 1 were. The observed -2.5 points sits inside an interval that
  reaches past -5: a loss greater than five points remains compatible with it, and neither a difference nor
  equivalence is established. Group 4 (dependent chains) shows the
  largest observed gap (18 vs 24 of 40); the registered analysis doesn't test groups separately.
- **Same inputs:** prompt-id hashes identical on 160/160. Full generated text is identical on 81/160 questions.
  Stage 2's nine divergent fixture runs first differed at an HF tie; this full-run comparison doesn't isolate the
  cause of each divergence.
- **Speed:** the latency rule (median S <= 2x median C and <= 60 s) holds; the observed speed-up is 1.68x at the
  median and 1.48x in summed time, one run each (repeatability not measured). The text bucket's rate is 1.69x C's;
  the chip bucket is similar (1,730 vs 1,712/s), as expected (the same executor). Inner buckets aren't comparable with outer times.
- **Memory:** the maximum observed combined footprint was 17,254 MiB (V100 UUID, both processes). There were no
  failed queries, but a 132-second startup gap (the runtime identity check) separates the initial after-load sample
  from periodic sampling; generation and shutdown were sampled about every 0.5 s (10,302 samples). No memory
  comparison with C is claimed. The startup's 30.6 s `load_seconds` is model load and warm-up, not total startup
  (the identity hashing came after it).
