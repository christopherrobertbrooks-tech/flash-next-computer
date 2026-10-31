# Making Qwenputer's computer faster on the V100 (research plan, 2026-10-31; nothing measured yet)

## The question
How much faster can Qwenputer's computer run on the V100 while staying exact, with every trace token either produced
or checked by the circuit itself?

## Where we start
- The circuit executor ran at **1,712 trace tokens/s** in the inside-4b run (chip-step time, checker excluded); a
  calculation like 123456789 x 987654321 is about 2,400 trace tokens, so ~1.4 s.
- Upstream reports ~3,200/s on an L4 (a newer, smaller card).
- The Python reference emulator does the same work at roughly **400,000 trace tokens/s** on one CPU core
  (2,146 tokens in ~5 ms): ~250x faster.

## How a step works today (from the source)
Per trace token, `computer.py` / upstream's `RvChat` do:
1. copy the token's embedding to the GPU;
2. replay one CUDA graph: the circuit's step as a chain of small Triton kernels, a few per sublayer over the 11
   circuit layers (sparse matrix rows, tiny);
3. copy the output code bits back to the CPU (`.tolist()`, a synchronization), decode them to the next token in
   Python, and look up its embedding.
The kernels are tiny, so the time is probably dominated by launch latency and the per-token CPU round trip, not
arithmetic. That is what we measure first.

## Candidate speed-ups
| | Idea | What stays true | Expected |
|---|---|---|---|
| A | **Keep the loop on the GPU**: decode the code bits and fetch the next embedding on the device, capture many steps in one CUDA graph, sync once per chunk | the circuit still produces every token | removes the per-token CPU round trip |
| B | **Batch**: run several machines side by side (Executor2 already has a batch dimension) | same | throughput up for many calls at once, not one call's latency |
| C | **Speculative execution**: the emulator drafts the whole trace (fast), and the circuit verifies it in **one parallel pass** (chunked prefill through the 11 circuit layers, as upstream's `gpucheck` already does), like speculative decoding | every token is still the circuit's own prediction (teacher-forced); the output is exactly what it would generate if all match | the biggest: prefill processes thousands of tokens per kernel launch |
| D | **Fuse the step into one kernel** (a "megakernel") | same as A | cuts launch overhead; heavy engineering |
| E | **Better firmware**: fewer instructions or trace tokens per calculation | changes the trace (needs new checks) | depends on the algorithm |

C changes the circuit's role from generator to verifier. That's legitimate (it's how speculative decoding works, and
the result is bit-identical when verification passes), but it has to be said plainly: the emulator does the stepping,
the circuit confirms it.

## Measurements (one V100 session, ~30-60 min, one job in the lock)
1. **Where the time goes:** upstream's `timed2` (the circuit's GPU time per step under a CUDA graph, no CPU decode) vs
   a full `Computer.call` per token: the gap is the CPU round trip (idea A's ceiling).
2. **Batching:** `timed2` at batch 1, 4, 16, 64: per-step time and throughput (idea B).
3. **Parallel verification:** teacher-forced chunked prefill of a real calc trace through the circuit (upstream's
   `gpucheck` prefill mode, bf16 HF modules, and/or the IR executor), timed and checked token by token (idea C).
4. Kernel count per step and the time per kernel (idea D's floor).

## Stop rules
- No change is kept unless it stays exact on the stage-3 gate (118 cases) and the shadow check.
- If the measurements show the arithmetic itself dominates (not overhead), A and D are dropped.
