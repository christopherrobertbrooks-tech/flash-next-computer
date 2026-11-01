# Speed session: where Qwenputer's time goes on the V100 (2026-10-31, 6:24-6:28 am)

One V100 session (clocks locked at 1,380 MHz by the lock tool). Raw: `speed/probe.json`, `speed/llama-bench.json`,
`speed/session.log`; script `speed/speed_probe.py`. Measurements only: nothing was changed. **Reviewed by Codex**
(review/codex-viewer-speed, blind 809ed9c, guided 2970e32): every raw number below reproduces; this page carries its
corrections to what they mean. These are component observations, not a tested speed-up.

## Summary (Codex's bounded wording)
The standalone teacher-forced step probe measured ~464 us per token, and a simplified host-read loop was ~4-5% slower.
That doesn't measure the full decoder's host overhead. Batching raised aggregate throughput, and the tested prefill
path was modestly faster than stepping. A separate cold/warm fla check found large compilation overhead, and a ~1.6x
warm slowdown in circuit prefill on one trace. llama.cpp's plain-model microbenchmark is promising but not matched to
the Hugging Face workload; how fast a Qwenputer port would run is unknown. Fusion and draft/verify are candidate
implementations that still need correctness and end-to-end measurements.

## The circuit (the computer), on standalone calculator traces (calc.elf: 1,111 / 4,248 / 14,041 trace tokens)
These probes build upstream's default standalone circuit (2,679 dims), not the merged chat circuit with its text
bridge; the live chat example is a different workload (2,430 chip tokens for the same multiplication vs 1,111 here).

| Measurement | Result |
|---|---|
| upstream `timed2`: wall time per teacher-forced step, one CUDA graph replay each, batch 1 (a Python loop with input copies, not GPU-only time) | **~460 us**, 2,142-2,178 trace tokens/s |
| the same plus a device-to-host read of `x[:64]` every token (a synthetic readback loop: no code-bit decode, token lookup or feedback) | 4.2-5.3% slower than `timed2` |
| our real runs (inside-4b's chip-step bucket, chat circuit) | ~1,712 tokens/s (an end-to-end observation under that bucket's definition) |
| batch 4 / 16 / 64 independent machines | 7,400 / 15,500 / **21,800** tokens/s total; a batch-64 step costs 6.3-6.4x a batch-1 step |
| teacher-forced chunked prefill of the whole known trace, bf16 HF modules | **2,560-2,850 tokens/s** (1.25-1.38x the readback loop); `hf.check`, outside the timer, exact on every IR dim at every position (19,400 in total) |

- **The 4-5% gap** is the incremental cost of that synthetic loop only. It's not a ceiling on moving the real decode
  onto the GPU (the real loop also decodes code bits, picks the next token and looks up its embedding).
- **1,712 vs ~2,150 tokens/s** (about 20% less throughput) is not an isolated bookkeeping cost: the circuit, the
  bridge, the kernel mix and the timing boundaries all differ between the two.

**Profile of one step** (torch profiler, batch 1): 36 kernels per step over 18 executor stages (8 head, 10 FFN, 18
"apply"). My first profile put apply at ~21%, but it wasn't saved. Codex's saved re-profile (10 steps): head 2,107
us, FFN 2,033 us, apply 462 us (**9.84%**), input copy 89 us, of 4,690 us total. Profiler time doesn't separate launch
latency from memory stalls, atomics or sparse arithmetic, so "latency-bound" is a hypothesis, not a finding.

## Text (the language model): not matched comparisons
| | tokens/s | conditions |
|---|---|---|
| plain Qwen3.5-4B, Hugging Face (inside-4b run) | ~37 | bf16, real questions, forced tool responses, varying history |
| Qwenputer-4B text steps, Hugging Face (inside-4b run) | ~28 | the same, widened model |
| plain Qwen3.5-4B, Hugging Face, PyTorch fallback / with fla | 44.1 / 48.2 | bf16, 256 greedy steps after one story prompt, Python argmax |
| plain Qwen3.5-4B, **llama.cpp** f16 GGUF (mainline 5e03bdd, sm70), `llama-bench` tg128 | **74.0** | f16 weights and caches, 128 tokens at depth 0 |
| the same, `llama-bench` pp512 (prompt reading, 512 tokens) | ~4,980 | not comparable with HF's 4,096-token prefill |

These differ in precision, prompt, length and history, so the ratios between them aren't an engine speed-up, and
none of them is Qwenputer in llama.cpp (it can't run there yet).

**If** a port made only the text steps faster, the inside-4b run's numbers (7,620.8 s total, 6,438.5 s in text steps)
give this range, holding everything else fixed:

| text steps faster by | total time saved |
|---|---|
| 1.5x | 28.2% |
| 1.68x | 34.2% |
| 2x | 42.2% |

Which multiplier a port could reach is unknown, and a different engine can also change outputs, trace work,
precision and memory.

## What it means
1. **Text is most of the time** (84.5% of C's question time in inside-4b), so it's the biggest lever if it can be sped
   up. The next useful measurement is a **matched plain-Qwen baseline** (llama.cpp vs Hugging Face at equal prompt
   ids, lengths, history, budgets and similar precision), then a correct Qwenputer port, then a full-system benchmark.
2. **Kernel fusion is not a small edit.** The head and FFN kernels accumulate with `tl.atomic_add` from several
   blocks, and apply then adds the sum to the residual, clears it, and (after heads) shifts lag history using the old
   input. Folding apply into the producers needs an ownership/reduction scheme or double buffers, since there's no
   barrier across blocks, and a new addition order isn't automatically bit-exact. Apply's measured share (9.84%) is
   only an upper bound on what deleting it could save, and chip steps were ~12.4% of inside-4b's question time.
3. **Batching helps throughput, not latency:** ~10x aggregate at 64 independent machines.
4. **"Emulator drafts, circuit verifies" is not disproven.** One HF prefill path gave a modest component gain
   (1.25-1.38x); drafting, a timed check, acceptance and fallback, and chat-state integration weren't measured. A
   cheaper idea than a full parallel scan: a **layer-major schedule** that batches the FFN and q/k/v/z projections
   over a known draft's positions and keeps only the head recurrence sequential, in fp32 on the existing IR. (Executor2's
   batch dimension is independent machines, not time, so it can't be reused as-is.) A candidate to measure.

## Next steps (Codex's recommended order)
1. Fix the viewer's committed-state, completion and cancellation contracts (on viewer-live: snapshot, queue and
   labels verified by Codex's re-check; the end-while-writing-a-call gap it found is fixed and confirmed, caefbc2).
2. A matched plain-Qwen llama.cpp baseline: the exact tool stream, budgets, prompt ids, similar numerics and caches.
3. The Qwenputer port in correctness milestones: circuit/Qwen path selection, separate positions and caches, firmware
   state injection, circuit-owned token choice, ownership and echoes. Unquantized first; reference traces and final
   state verified before any speed claim.
4. Then measure the full system and choose between native circuit kernels, a sparse executor, and layer-major verify.

## The "fast kernels" test (flash-linear-attention, 2026-10-31)
The viewer printed transformers' warning that Qwen3.5's gated-delta layers were running on PyTorch fallbacks
because `flash-linear-attention` (fla) and `causal-conv1d` aren't installed. My log filtering had hidden that warning
in every earlier run. fla 0.6.0 (with fla-core 0.6.0 and einops: 7.7 MB, pure Python and Triton) was installed with
`--no-deps` into a separate folder (`/mnt/nvme/qwenputer/fla-pkgs`, used only via PYTHONPATH; the venv is unchanged),
and `speed/fla_probe.py` ran with and without it (`speed/fla-fallback.json`, `speed/fla-fla.json`):

| | PyTorch fallback | fla 0.6.0 |
|---|---|---|
| plain 4B prompt reading, 4,096 random tokens (first use of that shape) | 994 tokens/s | 320 tokens/s (cold) |
| plain 4B decoding, 256 tokens | 44.1 tokens/s | 48.2 tokens/s |
| the circuit's teacher-forced prefill (4,248-token trace), my run (first use) | 2,841 tokens/s, exact | 221 tokens/s, exact (cold) |
| the same, Codex's re-check: 3 repeats | | 18.9 s cold, then **2.38 s** warm (~1,781 tokens/s), exact on all 4,248 positions |
| greedy replies to 5 pilot prompts | | 1 of 5 identical to the fallback's |

- My first timings included Triton's one-time compilation. Warm, fla was **~1.6x slower** than the fallback on that
  one circuit trace, not 13x. The plain-model prefill wasn't re-run warm.
- **Why** it's slower isn't shown: Volta's lack of native bf16 is context, not a demonstrated cause.
- The probe's kernel detector was broken (it always returned null), so which fla kernel ran wasn't recorded.
- 4 of 5 greedy replies differ, so outputs are sensitive to the kernel path. The cause wasn't investigated.
- **Keeping the fallback for that circuit case is reasonable.** This doesn't prove which path is faster for the plain
  model. `causal-conv1d` wasn't tried, and its share wasn't measured.
