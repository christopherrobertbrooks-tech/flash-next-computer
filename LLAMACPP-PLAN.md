# Qwenputer's language part in llama.cpp (plan v3, 2026-10-31; stage 3 registered after Codex's stage-2 review)

v2 carries Codex's plan review (review/codex-llamacpp-port 83904cc): the prompt-prefill path, new fixtures instead of
"historical" rows, a helper/state contract, a pre-registered stop rule, and corrected attribution.

## What you'll have at the end
Qwenputer answering maths questions on the V100 with its **language part running in llama.cpp** and its **computer
still on upstream's exact fp32 executor**, every computer token checked against the reference machine, watchable in
the live viewer, plus a measured answer: **how much faster than today's 16.8 s median per question, at what
accuracy**, on the same 160 questions as inside-4b. Good enough = 0 computer mismatches; the speed is whatever we
measure.

## Why this split (from reading upstream's chat loop)
**During generation after the prompt**, upstream's executor runs every token through the circuit, not the
checkpoint's computer layers. For a text token it builds the input to Qwen's 32 layers (`_exec_hc`) from two parts:
Qwen's own embedding dims and the 23 dims the circuit keeps (21 code dims, tau, mu), with everything else zero. Chip
tokens never reach Qwen's layers. So llama.cpp has to run **Qwen's 32 widened layers fed input rows directly**:
llama.cpp accepts float input rows instead of token ids, and `qwen35.cpp` uses its standard `build_inp_embd`.

**The prompt is different.** Upstream first runs the whole prompt through the full checkpoint, its bf16 computer
layers included. Only then does it snapshot the firmware and start the executor: `ExecRow.start` runs the prompt
from zero circuit state, then `load_init()` keeps the lag history. So the prompt's rows into Qwen's first layer come
from the bf16 computer layers, not from `_exec_hc`. Building prompt rows from the executor is a **candidate** that
stage 0 must prove equal to the HF rows (below), not an assumption.

The circuit stays out of llama.cpp on purpose, because none of llama.cpp's precisions is known to keep it exact:
- fp16 is impossible (computer weights reach 8.5e37);
- fp32 mispredicted 1 of 7,554 teacher-forced positions on the V100;
- bf16 on Volta in llama.cpp is unverified.

**What moves:** the Qwen-layer work being moved lies within the 84.5% text-step bucket. That bucket also holds the
executor step on text tokens, building `_exec_hc` and dispatch, which stay where they are, and the pipe adds work.
The moved share and the new boundary overhead will be measured; the speed gain is unknown.

## Stages (Codex review between stages; I check in with you at each end)
0. **Premise (one or two evenings; CPU plus short V100 jobs).**
   a. **New fixtures, captured on the trusted HF path.** These are new, not a replay of the saved inside-4b answers,
      which have no token ids or circuit values. Upstream's own generate (executor) runs on 12 fixed questions: the
      first 3 by id in each group (final-1-001..003, final-2-001..003, final-3-001..003, final-4-001..003). If none
      of them has two calls, a fixed two-call question is added. Hooks record, per position:
      - prompt and generated ids, and who owns the next token (text or circuit);
      - the text position;
      - the row entering Qwen's first layer: HF prefill rows for the prompt, `_exec_hc` rows after it;
      - HF's top-1 over the text vocabulary and its margin over the second choice;
      - hashes of the firmware, tokenizer, model and code.

      The fixtures are written once, never edited, and both engines are compared on them. They cover the prompt
      (tool syntax included), forced response bytes, the HALT newline and two successive calls. These 12 are also in
      the 160 of stage 3; this is a premise check, not tuning.
   b. **Prompt rows:** build the prompt's rows with the executor-based candidate, and compare them with the captured
      HF prefill rows elementwise, plus the first-token logits. Equal (bf16) passes. If they differ, the candidate
      fails and I report the options: keep the HF prefill, or run the 11 computer layers for the prompt elsewhere.
      No silent substitute.
   c. **Cut and convert:**
      - cut out layers 11-42 (renamed 0-31) with their norm compensation, the final norm, and the tied embedding
        sliced to the 248,320 text rows; the base tokenizer;
      - keep 3,968 input columns and the explicit head dims, layer types and rope settings;
      - convert to an f16 GGUF (~10 GB; 213 GB free) and load it in llama.cpp-mainline.
   d. **Teacher-forced comparison.** Feed the fixture rows into the llama.cpp model and compare the top-1 over the
      text vocabulary with the fixtures, at the same positions with the same mask.
      - **Control:** plain Qwen3.5-4B, HF bf16 vs llama.cpp f16, teacher-forced on the same 12 answers' text tokens.
      - **Hard checks, separate from agreement:** the row sent equals the fixture row bit for bit, positions match,
        and the state resets between questions.
      - **Stop rule, fixed now:** stop if the split's pooled top-1 agreement is more than 1.0 point below the
        control's, or if more than 2 of the 12 questions have a split disagreement where HF's margin is at least 1.0
        logit (not a near-tie).
      - **Denominator:** every position in the 12 fixed questions where Qwen chooses the next token: the prompt's
        last position and each generated position the circuit doesn't own. Positions the circuit owns (chip tokens,
        forced byte echoes, the HALT newline) are excluded. The control uses the same positions, the same text
        tokens before them and the same mask.
      - **The optional two-call question** (added only if none of the 12 has two calls) counts in the hard checks,
        not in the stop rule's agreement or its question count.
      - **Reported:** per-question disagreements, the first divergence and its margin.
      - Passing is a premise check, not proof the free-running answers match. Stages 2-3 still decide.
1. **Matched plain baseline (Codex's first step; one evening).** Plain Qwen3.5-4B f16 in llama.cpp, with the same
   RV32 emulator beside it:
   - the same prompts as token ids, the same tool frames (HALT newline included), the same budgets;
   - 12-question pilot, then all 160;
   - compared with inside-4b's R (130/160, 8.63 s median). It measures the accuracy and latency of the **combined
     engine-plus-precision change** (HF bf16 to llama.cpp f16), not either one alone.
2. **The split runner (one to two evenings).** Upstream's generate loop with Qwen's steps sent to a small llama.cpp
   helper program in this repo. The helper is built against llama.cpp-mainline's library; llama.cpp's source is
   unchanged.
   - **The pipe:** each message is framed with a length and a type: reset, prefill, step, end. The rows are f32,
     3,968 wide, with a finite-value check. The reply is the argmax over the text vocabulary, or an error.
   - **The helper's lifecycle:** a helper error or end of input is passed up as an error, never an answer. The child
     is cleaned up on every exit, and it runs under the parent's V100 lock and UUID, never taking the lock itself.
   - **CPU controls, before any GPU run:**
     - a fresh question resets the attention caches, the recurrent state and the text position, and A, B, A in one
       helper reproduces a fresh A;
     - chip-only spans don't advance Qwen, while forced byte echoes and the HALT newline do;
     - prefill and step-by-step histories agree on the same fixture;
     - a circuit-owned token (text or chip) overrides the helper's choice;
     - a mismatch, budget stop or early end can't count as a completed answer;
     - the prompt and `_exec_hc` rows equal the fixtures bit for bit.
   - The lock checks every chip token, as in inside-4b. The live viewer gets a `--backend llama` switch. 12-question
     pilot.
   - **Added after stage 0/1 reviews (Muse, Codex):** one short diagnostic run with `GGML_SCHED_DEBUG=2` (where each
     op runs, prompt and one-token paths; not timed); the per-token cost of the pipe; and, as a separate variant with
     its own GGUF hash, a **bf16 output head** (the source is bf16, so the stored values are identical; check the
     round trip bitwise, the placement, speed, and the stage-0 gate before using it). The baseline stays the f32 head.
3. **Full run (one evening, ~1-2 hours).** The registration below (v3, Codex stage-2 review c23fe76) is frozen
   before any stage-3 data; nothing in it changes after the run starts.
   - **Questions:** inside-4b's 160 (`inside4b/final-primary.jsonl`, sha256 `e3f98862352ad011...`), in file order;
     no exclusions, retries or substitutions.
   - **Arm S (primary):** the split runner (`llamacpp/run_split.py`) with the **bf16 output head** GGUF
     (`qwen-half-f16-bf16head.gguf`, sha256 `643e78a920c52bce...`). Chosen on development data (the stage-0 fixtures
     and the stage-2 pilot), before any stage-3 data: its stored values are bit-identical to the source, it picks the
     same argmax as the f32 head at all 14,745 fixture positions, and it passes the stage-0 gate. The f32-head GGUF
     stays the stage-0/2 reference; no full f32 run is part of this registration.
   - **Frozen identities:** `results/llamacpp/stage3-artifacts.json` lists the sha256 of every artifact (the GGUF,
     the helper and llama.cpp libraries, the runner, split_chat, the pipe client, the stage-0 gate and comparison
     scripts and data, inside-4b's run_qwenputer and frames, upstream's chipformer sources at their commit, the
     checkpoint's tokenizer, config, chipformer.json and embedding shard, the firmware, the questions): the
     **runtime** identity, which the runner checks at startup (`--expect`) and refuses to start on any difference.
     The **analysis** identity is frozen separately (Codex stage-2 confirm, finding 6) in
     `results/llamacpp/stage3-analysis.json`: the scorer (`stage5/score.py`) and its answer parser
     (`computer/stage4_requests.py`), the saved C replies and their question hash, the question file, the launcher
     (`llamacpp/stage3.sh`), the artifact builder and the runtime manifest, at a pinned repo commit;
     `llamacpp/stage3_analysis.py score` verifies them before it scores. Upstream is qwenputer `ca13310` with one local change, the qwenputer-v100 loading
     patch in `chat.py` (`results/llamacpp/upstream-chat-py.diff`: load straight to the GPU; Codex-verified then;
     SplitChat doesn't use that loader, and inside-4b's C ran with it). llama.cpp is mainline `5e03bdd` (sm_70 build).
   - **Preflight, in the same V100 session, right before the run:** the stage-0 split comparison with this exact
     helper and GGUF, then `stage0_gate.py` must PASS (this also rules out a helper that answers EOS or anything
     else wrong).
   - **Budgets:** 8,192 processed text tokens and 2,000,000 chip tokens per question; the lock's separate cap of
     2,000,000 machine records per call (a different unit, as in inside-4b); helper context 16,384 (the largest
     prompt is 726 tokens; the runner refuses a prompt that can't fit with the text budget). Greedy, the lowest id
     at a tie, EOS not fed, forced tokens as upstream, the computer on the executor only, no context shifting.
   - **Comparison:** S against inside-4b's saved C (`results/inside4b-full/full-C`), all 160 paired, scored with
     `stage5/score.py --compare S,C` (arm S added; unknown arms refused; inside-4b and stage 1 re-score identically):
     **S minus C**, Newcombe method 10 with the covariance correction, 95% interval, ±5-point margins; difference
     or equivalence only if the interval says so, else inconclusive. The four paired counts and every failure layer
     are reported. **Latency rule (as inherited, S-relative):** median S <= 2 x median C and <= 60 s; the speed ratio
     (median and summed time) is reported separately; no stricter speed target is registered.
   - **Timing:** each question's outer time (reset, prompt rows, prefill, generation; startup excluded and reported
     separately) and the inner chip/text buckets, which aren't compared with outer times. Clocks locked by the V100
     lock; no warm-up beyond the loader's.
   - **Memory:** S's sampled footprint by the method above (V100 UUID, both processes, through shutdown). **No
     memory comparison with C is claimed** (C's 18.98 GiB is PyTorch's allocated peak, a different quantity).
   - **Validity:** any lock error invalidates the design (the run stops, exit nonzero, "INVALID DESIGN"). A helper,
     protocol, non-finite-output or OOM error stops the run as incomplete. Every attempted row is kept. A text or
     chip budget stop or a wrong answer is a failed question, never a reason to rerun. If the run is interrupted by
     infrastructure (power, the machine, an unrelated crash), one restart from the beginning into a new folder is
     allowed and disclosed; no per-question reruns. Missing or duplicate ids withhold the verdict (the scorer
     refuses). Failed memory queries make the memory figure unavailable, not the accuracy result.
   - It's port-versus-original evidence (engine, precision and process split together), not an isolated engine
     effect, and not a native circuit in llama.cpp.
4. **Write-up and your call on sharing:** a private results page; an optional upstream Qwenputer issue (I draft, you
   approve; nothing to llama.cpp, whose policy bans AI-written posts).
5. *(Only if you want it after stage 4)* the circuit natively in llama.cpp: an exact executor op, or a precision
   proven exact.

## What I'm unsure of (stage 0 settles it)
- Whether the executor-built prompt rows equal HF's prefill rows (0b).
- Whether llama.cpp's multi-section rope for qwen35 matches HF's interleaved rope at text-only positions.
- Whether 32 layers at hidden 3,968 trips anything keyed off the layer count (`LLM_TYPE_UNKNOWN`).
- The helper's per-token overhead. If the pipe eats the gain, the fallback is llama-cpp-python, which needs your OK
  for a download and an sm_70 build.

## Rules
One V100 job at a time via the lock, never the 4070. No downloads expected. Work on branch `llamacpp-port`, merge on
your yes. Results stay private until your word.
