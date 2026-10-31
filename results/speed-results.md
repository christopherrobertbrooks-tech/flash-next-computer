# Speed session: where Qwenputer's time goes on the V100 (2026-10-31, 6:24-6:28 am)

One V100 session (clocks locked at 1,380 MHz by the lock tool). Raw: `speed/probe.json`, `speed/llama-bench.json`,
`speed/session.log`; script `speed/speed_probe.py`. Measurements only: nothing was changed.

## The circuit (the computer), on calculator traces (calc.elf: 1,111 / 4,248 / 14,041 trace tokens)
| Measurement | Result |
|---|---|
| upstream `timed2`: GPU time per step, one CUDA graph, batch 1 | **~460 us**, ~2,150 trace tokens/s |
| graph replay + a device-to-host read every token | ~485 us, ~2,060 tokens/s (the CPU round trip costs ~5%) |
| our real runs (`Computer.call`, inside-4b's chip bucket) | ~1,700 tokens/s (Python decode and bookkeeping add ~20%) |
| batch 4 / 16 / 64 machines side by side | 7,400 / 15,500 / **21,800** tokens/s total (a step only 6x slower at 64) |
| teacher-forced chunked prefill of the whole trace, bf16 HF modules | **2,560-2,850 tokens/s**, exact on every IR dim at every position (1,111 / 4,248 / 14,041) |

**Profile of one step** (torch profiler, batch 1): 36 kernels, ~500 us of GPU time: 8 head sublayers (~25 us each, 40%),
10 FFN sublayers (~20 us each, 39%), 18 "apply" kernels (residual += output, ~6 us each, 21%). The kernels are
latency-bound: at batch 64 a step is only 6x longer, so a single step leaves most of the GPU idle.

## Text (the language model)
| | tokens/s |
|---|---|
| plain Qwen3.5-4B, Hugging Face (inside-4b run, greedy decode) | ~37 |
| Qwenputer-4B text steps, Hugging Face (inside-4b run) | ~28 |
| plain Qwen3.5-4B, **llama.cpp** f16 GGUF (mainline 5e03bdd, sm70 build), `llama-bench` tg128 | **74** |
| plain Qwen3.5-4B, Hugging Face, short decode test (256 tokens), PyTorch fallback / with flash-linear-attention | 44 / 48 |
| the same, prompt reading (pp512) | ~4,980 |

## What it means
1. **The biggest end-to-end lever is the text side.** In the inside-4b run, Qwenputer spent 6,439 s in text steps and 944
   s in computer steps. llama.cpp runs the plain 4B's text 2x faster than our runs' Hugging Face speed (about 1.5x
   faster than the best Hugging Face manages on this card, 48 tokens/s; see the kernel test below), so a llama.cpp
   port of Qwenputer (Codex's route 2) would cut its total time by about a third to a half. That's an estimate from the plain model: the widened
   Qwenputer model isn't measured in llama.cpp, because llama.cpp can't run it yet.
2. **The circuit's time is its kernels, not the CPU.** Removing the per-token round trip gains ~5% (more, ~20%, if the
   Python decode also moves onto the GPU). Folding the 18 apply kernels into their producers would cut ~20%. Using the
   GPU better inside each kernel (more parallel work per head/FFN step) might give ~2x more; that is kernel engineering,
   and every change must stay exact (stage-3 gate, upstream `check`, the live shadow).
3. **Batching helps throughput, not latency:** 10x at 64 machines, useful when many calculations run at once.
4. **"Emulator drafts, circuit verifies" doesn't pay on this card yet.** The parallel prefill was barely faster than
   stepping (2.6-2.9k vs 2.15k tokens/s), most likely because the circuit's recurrent head updates run as step-by-step
   PyTorch fallbacks on the V100 (no fused parallel kernel). It would need a parallel-scan kernel written for the circuit's
   heads, a research project of its own. It did stay exact on all 19,400 positions, so the route is correct, just slow.

## Ranked next steps
| | Step | Gain (estimate) | Work |
|---|---|---|---|
| 1 | Qwenputer in llama.cpp (Codex's route 2: two paths, separate positions, firmware state injection, circuit-owned output) | ~2x end-to-end | a multi-stage project |
| 2 | Fold the apply kernels into the head/FFN kernels | ~20% of circuit time | small, in upstream's Triton code |
| 3 | Move token decode + embedding onto the GPU, many steps per graph | ~20% of circuit time | small to medium |
| 4 | More parallelism inside the head/FFN kernels | up to ~2x circuit time | medium to large |
| 5 | A parallel-scan kernel for prefill verification | large, if it works | research |

## The "fast kernels" test (flash-linear-attention, 2026-10-31)
The viewer printed transformers' warning that Qwen3.5's gated-delta layers were running on PyTorch fallbacks
because `flash-linear-attention` (fla) and `causal-conv1d` aren't installed. Our own log filtering had hidden that
warning in every earlier run. fla 0.6.0 (with fla-core 0.6.0 and einops: 7.7 MB, pure Python and Triton) was
installed with `--no-deps` into a separate folder (`/mnt/nvme/qwenputer/fla-pkgs`, used only via PYTHONPATH; the
venv is unchanged), and `speed/fla_probe.py` ran with and without it (`speed/fla-fallback.json`, `speed/fla-fla.json`):

| | PyTorch fallback | fla 0.6.0 |
|---|---|---|
| plain 4B prompt reading, 4,096 tokens | **994 tokens/s** | 320 tokens/s |
| plain 4B decoding, 256 tokens | 44.1 tokens/s | **48.2 tokens/s** |
| the circuit's teacher-forced prefill (4,248-token trace) | **2,841 tokens/s**, exact | 221 tokens/s, exact |
| greedy replies to 5 pilot prompts | | 1 of 5 identical to the fallback's (different arithmetic order) |

fla's kernels run correctly on the V100 but compute in bf16, which Volta lacks in hardware. So they're 3-13x slower
for prompt reading and parallel verification, and only 9% faster for decoding. **The fallback is the better choice on
this card**: the earlier runs weren't crippled by the missing kernels, and the parallel-verify route still needs a
kernel written for Volta (fp32 or fp16), not fla. `causal-conv1d` wasn't tried (it needs a build for sm_70 against
torch 2.14; its share of decode time is small).
