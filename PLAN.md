# Flash-Next Internal Computer — Working Project Plan

Updated October 30, 2026. The BF16-module arithmetic run is complete; integration remains untested.

## Goal

Make a deterministic, programmable computer work with Qwen3.8-Flash-Next Coder through Strata on Chris's Tesla V100, then test whether it improves useful tasks compared with the same model alone and with an ordinary host Python tool. The Flash-Next model is the target; Qwen3.5/3.6 experiments are testbeds. Keep the language model's normal behavior and useful speed while preserving the computer's correctness.

## What we know

- On the V100, the Qwenputer BF16 Hugging Face computer modules matched the reference circuit on nine selected programs: 7,554 positions in reference-fed prefill and in decode. This is not the full official RISC-V suite.
- The full Qwenputer 4B arithmetic run scored 195/200. Its merged model was loaded in BF16; the separate Triton computer executor runs circuit IR arithmetic in FP32. The run report says 200 records, 2,208,347 computer tokens, zero lock errors and zero head mismatches. Codex independently examined 160 records; the other 40 were verified by the project's later verifier but could not be individually downloaded through Codex's connector. Scores do not prove an improvement over the author's results; their problem-level pairing is unverified.
- A direct BF16-to-FP16 cast is not a sound way to preserve this checkpoint: many computer and embedding values overflow. This is a static result, not proof that a recompiled, rescaled, or mixed-precision design cannot work.
- The BF16 module checker run as FP32 has a teacher-forced next-token prediction mismatch on one V100 prefill case. A follow-up locates its first position; cause remains unproven. Do not confuse this path with the separate FP32 IR executor that passed its recorded lock checks.
- The completed BF16-module run reported 195/200 final answers, zero lock errors or head mismatches, and identical generated text to the executor on all 200 cases. Codex and a separate reviewer independently matched text and token IDs for 160 paired records; the remaining 40 were not independently retrievable. Equal generated text does not prove identical hidden states. For those 160 cases, reported chip time was about 217 versus 1,704 trace tokens/s (executor approximately 7.9x faster). This measures computer time, not end-to-end speed. [Completed-run review](https://github.com/christopherrobertbrooks-tech/qwenputer-v100/blob/review/codex-step4/REVIEW-STEP4-CODEX.md).
- Upstream has a compact sparse circuit executor and describes padding reduction; these are prior art, not our new invention. Flash-Next integration remains unproven.
- In the October 18 Strata 0.1.41 Coder IQ1_M comparison, startup readings were 32,007 MiB with vision (11,642 expert slots), 31,833 MiB without vision (12,288 slots), and 30,645 MiB without vision or MTP (12,288 slots). These are startup readings, not peak memory or a full allocation breakdown. [Raw log](https://github.com/christopherrobertbrooks-tech/strata-vs-llamacpp/blob/main/results/2026-10-18-round2/sweep.log).
- The sharp performance decline at reduced expert residency came from the comparison's **llama.cpp** cache sweep, not a controlled Strata sweep. It suggests a risk on this PCIe x4 link; the impact of reserving VRAM for the computer must be measured in Strata itself.

Evidence: [project report](https://github.com/christopherrobertbrooks-tech/qwenputer-v100/blob/fp16/REPORT.md), [independent review](https://github.com/christopherrobertbrooks-tech/qwenputer-v100/blob/review/codex-v100/REVIEW-FINDINGS.md), [Strata comparison](https://github.com/christopherrobertbrooks-tech/strata-vs-llamacpp), [Qwenputer source](https://github.com/divagr18/qwenputer).

## Public Strata source mapping (October 29, user-supplied)

This mapping is from the public source as reported by Chris, not an inspection of the deployed gateway. Version, values, and absence-of-hook findings must be checked against the pinned code and deployment before implementation. The cited public release is v0.2.8; setup's minimum separately fetched engine is 0.1.46. Pin the repo tag/commit and actual engine --version independently, along with the config, binary, tokenizer and launch path.

- Flash-Next Coder IQ1_M has 256 of the full model's 512 experts per layer; the full model has 48 layers and 10 selected experts per token. The reported Coder download is 58.4 GB and expert arena 23.4 GB. Arena size is not the full VRAM budget.
- The source mapping identifies GDN recurrence and convolution history, QSA KV pages, indexer keys and four hyper-connection streams as relevant model state. Prefix-keyed caches and snapshots do not provide a separate internal trace-token namespace or fence. Verify the precise per-layer mix for the chosen engine.
- The server's existing tool flow emits tool-call text, invokes tools outside the engine and reinserts tool results as ordinary conversation tokens. No configuration-only hook or trace-token injection/isolation API was found in the supplied mapping. Validate against the chosen commit before calling this exhaustive.
- MTP and DFlash2 verify drafts against the target; feeding computer traces through Flash-Next could contaminate cache, recurrent state, positions and draft inputs. Disabling drafting does not itself isolate state. The cited MTP (~1 GB) and DFlash2 (~0.8 GB buffers) figures are source-specific estimates, not measured gateway allocations.
- KV modes and streaming, adaptive expert cache, a 700 MiB reserve and the SSD n-gram table affect the budget. The source mapping reports an int8 KV footprint near 13.7 KB/token and approximately 1.7 GB at 128K, plus streaming above 64K; establish units, allocation and actual deployed settings before using those estimates. The n-gram table is on SSD; do not budget the entire file as mandatory VRAM. BF16-to-FP16 safety claims for Flash-Next weights do not establish exactness for Qwenputer.

### Integration fork

1. **Server-side bridge:** use the existing external tool flow to call a reference-checked computer executor and return results as prompt tokens. This is a controller plus executor, not a merged-weight model. It provides a useful early baseline against host Python.
2. **Engine fork:** add a computer execution path with explicit memory ownership and isolation from Flash-Next's cache, token positions, recurrent state and speculative verification. This is the candidate for deeper in-engine integration; its design and feasibility remain unproven. A separate executor housed in the engine is not automatically a computer compiled into Flash-Next's weights; describe what is actually implemented.

After Claude's run, perform read-only gateway checks: repo tag and engine version; full run config and launch path; actual model, context, KV type and streaming, expert cache mode, resident/mmap policy, VRAM reserve, MTP/DFlash2 and KVA settings; idle and loaded VRAM/RAM; whether parallel slots or MCP are used. Measure the compact executor's own footprint and evaluate expert displacement under matched settings. Do not interfere with the running tests.

## Design hypothesis

Keep Flash-Next's existing quantized model and MoE execution at their native dimensions. Run a compact, reference-checked computer path only when needed, and bridge requests and results without feeding internal execution trace tokens through the full model. Preserve the computer state across its own steps and keep its precision and numerical behavior independently testable. This is a hypothesis: it is not yet a working integrated model, and using a separate runtime component may differ materially from a single merged-weight checkpoint. Explicitly report the implementation and how the computer is represented.

Do not assume Flash-Next shares Qwen3.5's state implementation or that its tokenizer accepts Qwenputer trace tokens unchanged. Do not assume a hot expert or physical RAM offload solves exact computation. Don't assume the compact executor runs without an external driver; isolate that distinction.

## Stages and decision gates

### 1. Finish and review the baseline

- Completed: Claude's BF16-module arithmetic run and independent review of 160 paired records. Obtain and strictly compare the remaining 40 saved multi-step records before calling all 200 independently verified.
- Independently compare it with the existing executor run. Separate exact internal trace, numerical agreement, and final-answer accuracy.
- Confirm per-test exits, unique records, CPU/GPU timing boundaries and warmup. Record failures and stop conditions.
- Gate: a reproducible bounded computer path, or a clearly diagnosed failure. A failure changes the implementation route; it does not end the project.

### 2. Map the real Flash-Next integration

- Pin the actual deployed Strata source/version and Coder model/pack, tokenizer, memory settings and GPU cache layout.
- Trace Flash-Next's mixer, state and context caching, tokenizer and output head, speculative decoding, and how the engine handles tool messages and internal state. Compare with Qwenputer's firmware, token encoding, state lifecycle, output bridge, and exactness requirements.
- Write a compatibility table: existing support, required adaptation, unknowns, and measurements. Identify whether a Flash-Next implementation can stay in model execution or uses a distinct executor and bridge; don't call the latter a single compiled-weight model.
- Produce a byte budget: current model, resident experts, KV and recurrent state, computer weights and state, temporary workspace and margin. Identify current VRAM and host-RAM limits, not just checkpoint size.
- Use the October 18 startup readings as historical baselines, not current free-memory guarantees. Measure total and available VRAM, peak allocation, computer footprint, expert residency, cache misses and speed together under matched vision, MTP and context settings. Include both normal and reduced-feature configurations; quantify the effect of reserving memory before selecting an integration budget.
- Make trace isolation an early proof: test a small pause-and-resume sequence for unchanged prompt cache, token positions, model state and MTP verification before building full integration.
- Gate: a specific implementation with testable interfaces and a credible fit in 32 GB VRAM and 16 GB RAM; otherwise reduce scope or revise layout.

### 3. Interface proof with the existing 4B computer

- First candidate: the compact executor, supported by matched outputs on 160 independently inspected cases and approximately 7.9x faster reported computer time on those cases. Reproduce a small set of arithmetic and trace cases and establish a frozen baseline before changing anything. Measure the executor's isolated and concurrent GPU/host memory and state footprint before attaching it to Flash-Next.
- Prototype the message and state handoff against a small stand-in or saved Flash-Next tool-call traces. Validate valid calls, invalid syntax, multi-call sequences, state reset, long-running/budget stops and exact returned bytes.
- Compare every computer step with the reference on a declared corpus. Keep an untouched path as control. No claim of a Flash-Next-internal computer until integrated.
- Gate: zero observed trace mismatches on declared tests, correct call/result routing, and measured peak VRAM, RAM, latency and behavior when the computer is idle.

### 4. Integrate with Flash-Next and Strata

- Work in an isolated branch or private prototype, not the live server. Preserve existing expert dimensions and quantization where possible; avoid materializing zero padding into every expert.
- Define context isolation: computer trace tokens and state must not silently alter Flash-Next's prompt cache, token positions or MTP verification. Test interruption, restart, and consecutive conversations.
- Verify executor/kernel precision separately on V100. Check the GPU cache and expert residency before and after adding the computer.
- Gate: baseline text outputs remain acceptably comparable, reference checks pass, no uncontrolled memory growth, and no reliance on streaming gigabytes of active weights from disk or RAM.

### 5. Measure usefulness

Compare the same Flash-Next build, model, prompts and sampling across:
1. Flash-Next alone.
2. Flash-Next with a host Python calculator or sandbox.
3. Flash-Next with the proposed computer path.

Use held-out arithmetic, small executable checks, multi-step questions, and normal non-computer requests. Predefine scorer, tests and stop rules. Measure final answer correctness, computer reference mismatches, tool-call correctness, loops, wall-clock time, time to first token, trace-token throughput with timing definition, GPU/host peak memory, expert residency and cache traffic. Run enough paired cases to distinguish direction from noise; classify failures rather than asserting causation from a few examples.

Predicted advantage to test: retaining exact state across many small operations might reduce repeated handoff overhead or improve state retention compared with repeated host-tool calls. This is unproven; include an efficient persistent host-Python baseline and measure end-to-end accuracy and latency before claiming an advantage.

Success means a useful measured advantage or a clear new capability, with a practical latency and memory tradeoff. Even a correct internal computer may be slower than host Python. If so, document that result and decide whether the scientific value justifies further work.

## Constraints and working rules

- V100 32 GB, 16 GB host RAM, PCIe Gen3 x4, CUDA 12.x. The RTX 4070 may be used temporarily for an experiment but nothing permanent; record duration. Approximately 300–500 GB disk can be made available if needed, but current free space is unverified; do not clear it automatically.
- Avoid Docker and hardware upgrades. No model downloads, public posts or merges without Chris's approval. Don't interrupt Claude's live run or alter its files.
- Keep blind and guided review separate; use a fresh reviewer at major gates. Compare claims to raw logs, and state what's missing.
- When the project feels stalled, identify the smallest meaningful test, record what it would prove, and decide whether to change approach. A failed design is useful evidence; it is not proof the goal is impossible.

## Next action

Independently verify the remaining 40 paired multi-step records. Use the public-source mapping above as a starting point; after the run, inspect the actual gateway deployment and complete its memory budget. Do not build or launch the integration until the result and resource budget are understood.

