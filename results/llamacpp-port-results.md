# Qwenputer's language part in llama.cpp: results (2026-10-31)

Builds on [Qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal (MIT); please cite it as given in
the [README](../README.md#credits-and-citation).

**What you have:** Qwenputer-4B answering maths questions on the V100 with its **Qwen half running in llama.cpp**
and its **computer on upstream's exact fp32 executor**. Every computer token is checked against the RV32 reference
machine, and you can watch it in the live viewer (`viewer/rvlive.py --backend llama`). The 18.8 GB HF checkpoint isn't
loaded.

**What it measured** (registered stage 3, 160 questions, one run each):

| | Split (llama.cpp + executor) | Original Qwenputer (HF, inside-4b) |
|---|---|---|
| right answers | **131/160** | **135/160** |
| S minus C, 95% interval | **-2.5 points, -8.8 to +3.7**: inconclusive | |
| chip trace tokens matching the RV32 reference | **all 1,552,141** (0 lock errors) | all 1,615,594 |
| median question time | **10.0 s** | 16.8 s (**1.68x**) |
| summed question time | 5,148 s | 7,621 s (**1.48x**) |
| text-step rate | 48.0 tokens/s | 28.4 tokens/s |
| computer-step rate | 1,730 tokens/s | 1,712 tokens/s |

The accuracy result establishes neither a difference nor equivalence: a loss greater than five points is still
compatible with the interval, as is a small gain. The speed-up is the observed one in this comparison. Engine
(llama.cpp vs HF), precision (f16 Qwen layers and a bf16 head vs bf16 throughout) and the process split changed
together, and timing repeatability wasn't measured.

Plan: `LLAMACPP-PLAN.md` (v1 to v3). Every stage was reviewed by Codex in two passes (blind, then guided), and its
corrections were applied before the next stage. Stage pages: `results/llamacpp-stage0.md` to `-stage3.md`.

## How it works
Upstream's own chat loop already runs every token through the circuit on its fp32 executor and builds Qwen's input
row from two parts: Qwen's embedding dims and the 23 dims the circuit keeps. Chip tokens never reach Qwen's layers.
So llama.cpp only has to run Qwenputer's 32 widened Qwen layers (hidden 3,968, their norm compensation included),
fed those rows as input embeddings:
- **The cut:** layers 11-42 renamed 0-31, the final norm, the tied embedding sliced to the 248,320 text rows; all
  426 tensors bit-identical to the source.
- **The GGUF:** f16 layers. The output head stays out of f16 (it holds values up to 2^40, the final norm up to 2^60,
  beyond f16's range): f32 in stages 0-2 and **bf16** in stage 3. The bf16 head is a bit-exact copy of the source
  values, 10.7% faster and 1.8 GiB smaller.
- **The helper:** `llamacpp/qwen_half_helper` (built against mainline llama.cpp's library, which is unchanged) over
  a binary pipe. It fails closed on errors, non-finite output, bad replies and timeouts.
- **The runner:** `llamacpp/split_chat.py` (upstream's generate copied line for line, only the two Qwen calls moved)
  and `llamacpp/run_split.py` (inside-4b's row format; a lock error stops the run as an invalid design).

## Stages
| Stage | Question | Result |
|---|---|---|
| 0 premise | Does the cut Qwen half in llama.cpp pick the same next token as HF, given HF's exact input rows? | **PASS**: 99.782% greedy agreement over 13,746 Qwen-chosen positions vs a plain-Qwen precision control of 99.796%; every disagreement at a near-tie. The executor-built prompt rows equal the HF prefill rows bit for bit. A fail-closed gate with negative controls. |
| 1 baseline | Does llama.cpp speed up plain Qwen with the emulator beside it? | 129 vs 130/160 (inconclusive); median 4.67 vs 8.63 s (1.85x). |
| 2 split runner | Does the split run end to end with the computer exact? | 12-question pilot: all EOS, 0 lock errors in 188,945 chip tokens. **History oracle:** along the split's own histories, all 23,956 rows sent to llama.cpp equal upstream's independent HF chip-layer path bit for bit, and all 189,771 circuit choices agree. |
| 3 full run | Registered comparison with inside-4b's original Qwenputer | the table above |

## Protocol and identities
- **Hardware:** Tesla V100-PCIE-32GB (UUID `GPU-9c9e042a-...`), clocks locked at 1,380 MHz by the lock tool, the same
  for C and S. Gateway: i7-13700KF, 16 GB RAM.
- **Software:** llama.cpp mainline `5e03bdd` (sm_70 build, CUDA 12); upstream qwenputer `ca13310`, with one
  disclosed local change (`results/llamacpp/upstream-chat-py.diff`: the earlier qwenputer-v100 patch that loads the
  HF checkpoint straight to the GPU; SplitChat doesn't use that loader, and inside-4b's C ran with it); torch
  2.14.1+cu126.
- **Frozen before stage 3:**
  - the runtime identity, `results/llamacpp/stage3-artifacts.json`: 106 files, checked at startup; all matched;
  - the analysis identity, `results/llamacpp/stage3-analysis.json`: the scorer, its parser, the saved C replies,
    the questions and the launcher, verified before scoring.
- **Preflight** in the same V100 session: the stage-0 comparison with the exact helper and GGUF, gate PASS.
- **Questions:** inside-4b's 160 (sha256 `e3f98862...`), in order. Prompt ids identical to C's on 160/160.
- **Budgets:** 8,192 text and 2,000,000 chip tokens per question; the reference's separate 2,000,000-record cap
  per call.
- **Scoring:** `stage5/score.py --compare S,C`, with Newcombe method 10 and a ±5-point margin. Inside-4b and stage 1
  re-score identically with the extended scorer.
- **Reproduce stage 3** (gateway, inside the V100 lock):
  `llamacpp/stage3.sh results/llamacpp/stage3-artifacts.json OUT`. Score on Dev-Console with
  `python llamacpp/stage3_analysis.py score results/llamacpp/stage3-analysis.json S_DIR OUT.json`, where S_DIR holds
  the run's replies.jsonl (it's gzipped in `results/llamacpp/stage3/`).

## Caveats (what this is and isn't)
- **One run per configuration.** The speed ratios are observed, not repeated. The answers diverge on 79/160 questions,
  so per-question work differs; S produced more text tokens in total (200k vs 183k).
- **Not an isolated engine effect.** Engine, precision and the process split changed together. The f16 Qwen layers
  differ from the original bf16. Stage 0 bounds the effect teacher-forced; the full run can't attribute each
  divergence.
- **The bf16 head was chosen on development data** (the stage-0 fixtures and the stage-2 pilot) before any stage-3
  data. Unchanged stored values and fixture agreement don't prove equal free-running outputs.
- **The circuit isn't in llama.cpp.** It runs on upstream's fp32 executor; the checkpoint's bf16 computer layers
  aren't executed in this benchmark (the stage-2 oracle shows both upstream paths agree on the tested histories).
  A native llama.cpp circuit would need a precision proven exact: f16 can't hold its weights, fp32 mispredicted 1 of
  7,554 positions in earlier work, and bf16 on Volta in llama.cpp is unverified.
- **"All chip tokens matched"** is the observed stream on these questions, not a proof for every input, and not a
  claim that the answers or tool requests were right.
- **Memory:**
  - The split's maximum sampled footprint was 17,254 MiB, both processes together.
  - There is a 132 s startup gap in the sampling (the identity check); generation and shutdown were sampled every
    0.5 s.
  - It isn't compared with C: C's 18.98 GiB is PyTorch's allocated peak, a different quantity.
- **Group 4** (dependent chains) shows the largest observed gap (18 vs 24 of 40). This is descriptive; no registered
  subgroup test was run.

## Possible next steps (not planned)
- A pre-registered timing-repeatability run, both configurations counterbalanced.
- Trimming the output head to the 2,560 language columns: the computer's columns add the same offset to every
  text logit (Codex), so it's an optimization candidate with its own exactness check.
- A native llama.cpp circuit, only with an exact precision.
