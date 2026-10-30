# Strata v0.1.41 (fb58e0d) map for Flash-Next Coder: research agent report, 2026-10-30

Read-only: source in `/mnt/steam/Strata-0141`, pack files, the GGUF header (metadata only, stdlib parser), logs.
Nothing launched or changed. UNCONFIRMED marks items that need a run.

## 1. Architecture (GGUF `general.architecture = qwen4exp`, "Qwen3.8-Flash-Next GSQ-RCO Coder")

| Item | Value | Source |
|---|---|---|
| Layers | 48 (`qwen4exp.block_count`) | GGUF; guard `gguf_reader.hpp:575-604` |
| Mixers | 36 Gated DeltaNet + 12 "QSA" full-attention at 3, 7, ..., 47 (`full_attention_interval 4`, `compress_ratios` 4 there) | `layout.hpp:30,63-65` |
| GDN | state 128, 16 k-heads / 48 v-heads, conv kernel 4, conv channels 10240, inner 6144 | `layout.hpp:32-38` |
| QSA | 24 q heads / 2 KV heads, head dim 256, rope dims 64 (mrope 11/11/10), freq_base 1e7; indexer 4 q heads, one shared 128-dim key, block 4, top_k 2048 | GGUF; `qsa.hpp:103,112` |
| Hidden size | 2560 | GGUF |
| Hyper-connections | 4 streams, low rank 320; residual R = 4 x 2560 = 10240 floats | GGUF; `session.hpp:47`; `layout.hpp:47-49` |
| MoE | 256 experts (pruned from 512), top-10, width 640, + 1 shared expert (640) with sigmoid gate | GGUF; `generate.cpp:2498-2499` |
| PLE | per-layer n-gram embedding on layer 1: 3-gram hash, 16 heads, 320,001,536-row table in shard 2, read from SSD | GGUF `ple.*`; log |
| Vocab | 248,320; specials 248044-248076; `<|im_end|>`=248046 EOS; 243 unused `[PAD248077..248319]` (type 5) | GGUF |
| MTP layer | separate `mtp/rt`; router 512x2560 (all 512 experts) | `mtp/rt/dense.txt`; `mtp.hpp:1-21` |

## 2. Per-sequence state (`session.hpp:31-96`, `session.cpp:40-74`)

| State | Format | Size |
|---|---|---|
| GDN recurrence + conv history | fp32 | (128*48*128 + 10240*3) x 4 B = 3.27 MB/layer; x36 = ~118 MB |
| KV (int8) | int8 + fp16 scale per 64; paged | 1,056 B/cell/QSA layer; x12 = 12,672 B/token |
| Indexer | fp32 pooled keys | ~128 B/token/layer, ~1.5 KB/token |
| RoPE table | fp32 once | ~32 MiB at 131072 |
| PLE | prev ids + rows of 10240 floats | small |
| MTP drafter | own int8 KV ring (window 32768) + draft head | log: draft head 212.9 MiB |

All carved once at `--max-context` (`session.hpp:98`). At 131072: ~1.55 GiB KV + 0.2 GB indexer + 0.12 GB GDN +
scratch, ~2 GB (UNCONFIRMED).

## 3. Prefix caching, snapshots, injection

| Mechanism | Detail | Ref |
|---|---|---|
| Engine protocol | stdin `GEN <max_new> [key=val...] id,id,...` (`GENI` + image file); stdout `T <id>`, `DONE`; `STOP`, `QUIT`, `SAVE`/`RESTORE <path>` | `generate.cpp:5660-5676, 8708, 8910-8964, 7960` |
| Reuse | resume from the live sequence if the new prompt extends it exactly, else the longest prefix checkpoint | `generate.cpp:9124-9131` |
| Checkpoints | at the last turn boundary, every 16,384 new prompt tokens, a system-prompt root at 2048; 6 in RAM (~118 MB each); GDN/PLE/indexer state, KV stays | `generate.cpp:597-601, 732-760`; `conversation_cache.hpp:10-30` |
| `pin=N` | shared read-only prefix | `generate.cpp:8946-8953` |
| Snapshots | `POST /slots/0?action=save|restore` | `docs/DETAILS.md:951-990` |
| Hidden-token injection | no public API; the server does it internally (reasoning wrap-up: STOP, `prompt = prompt + seg + extra`, re-GEN on the live prefix) | `server.py:3621, 3631-3654` |
| Steering | `--cvec` adds/projects the residual after a layer's FFN | `cvec.hpp`; `layer.cpp:1487-1491` |

GDN state can't be truncated: reuse only at an exact live end or exact checkpoint.

## 4. Speculative decoding

| Item | Detail | Ref |
|---|---|---|
| Verify window | last accepted + drafts through all 48 layers; bit-identical to greedy | `verify.hpp:1-20` |
| Rollback | window appends K/V + indexer keys, not GDN; `commit(n_keep)` replays accepted tokens into GDN, restores indexer tail, resets PLE history; rejected cells overwritten later | `verify.hpp:13-18, 142` |
| Sampling | exact-match acceptance by default; rejection sampling opt-in (`STRATA_SPEC_PROB=1`) | `spec_prob.hpp:1-30`; `generate.cpp:10681` |
| Drafters | MTP (Q8_0), suffix/prompt-lookup (`--suffix-draft 3`) | `mtp.hpp:1-21` |
| DFlash | absent in this version | grep |

## 5. Tool calling (`serve/`)

| Item | Detail | Ref |
|---|---|---|
| Format | XML `<tool_call>` (248058) ... `</tool_call>` (248059) with `<function=NAME>` and `<parameter=P>` | template; `frontend.py:9, 509-515, 665` |
| Detection | Python `OutputParser` over the detokenized stream; `tool_start`, `tool_call` events; malformed-call recovery | `frontend.py:698-1184` |
| Results | `<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>`: ordinary tokens | template |
| Server-side execution | `run_with_mcp`: MCP tools run in the server, call + result appended, re-rendered, re-GEN (up to 8 rounds); opt-in (`"strata_mcp": true`), only from the server's own web page | `server.py:3829-3922, 4952-4970`; `mcp.py:1-38` |
| Endpoints | OpenAI, Anthropic, Responses; no raw-token `/completion` | `server.py:1-20` |

## 6. Memory budget

| Item | Value | Source |
|---|---|---|
| Auto-cache | slots = (free - (reserve + prefill) - draft head) / largest blob, capped at profile length; then re-sized per layer | `generate.cpp:4305-4347, 4469+` |
| Dense weights | 1406 MiB pack + 2018.88 MiB native projections + Q5_K head 521,472,000 B | log |
| MTP | 950 MiB + 218 MiB draft head | log |
| Verify buffers | 75.1 MiB | log |
| Embedding | IQ4_XS 322 MiB, mapped host RAM | log |

| Engine start | Config | Auto line | Cache | Free after load |
|---|---|---|---|---|
| 10-18 11:27:23 am | vision + MTP | 23.09 GiB free, 700 MiB reserved (+218 draft head) -> 8949 | 11642 slots, 22.19 GiB; 4.63 GiB pinned RAM | 487 MiB |
| 10-18 12:17:48 pm | no vision, MTP | 24.49 GiB free -> 9513 | 12288 slots, 23.42 GiB (all resident) | 661 MiB |
| 10-18 12:19:08 pm | no MTP, `--spec 2` | 25.42 GiB free (+0 MiB) -> 9976 | 12288 | 1849 MiB |
| 10-23 6:55:04 am | vision + MTP (current) | 22.99 GiB free -> 8909 | 11591; 4.73 GiB pinned RAM | 485 MiB |

Vision costs ~1.40 GiB (separate `strata-vision` process). Less VRAM = fewer resident slots; misses run on the CPU
AVX2 pool from the pinned RAM copy. Prefill borrows 1767 slots (3.40 GiB). Host RAM: 4.6-4.7 GiB pinned experts +
322 MiB embedding + ~90 MB PLE rows + checkpoints ~118 MB each.

## 7. Precision

| Part | dtype | Ref |
|---|---|---|
| Residual / activations | fp32 (q8_0/q8_K for quantized GEMVs) | `layer.hpp:66-120` |
| KV | int8 + fp16 scales | `layer.hpp:209-247` |
| GDN state | fp32 | `session.hpp:35` |
| Norms, ssm_a/dt/conv | fp32 | index.txt |
| HC, routers, indexer q/k, ssm_alpha/beta, PLE k/v | 16-bit bf16 widened to f32 by bit shift (no bf16 hardware needed) | `bf16_gemv.hpp:1-30`, `weights.hpp:41-46` |
| Other projections | IQ quants; MTP Q8_0 | |

A small fp32 module could sit after the FFN write (the cvec hook, `layer.cpp:1487`) or server side.

## 8. Tokenizer
GPT-2 byte-level BPE (`pre qwen35`), 247,587 merges. 243 unused PAD ids could be named without retraining (their
embedding/head rows probably untrained, UNCONFIRMED; the PLE hash would also see them). `<think>`, `</think>`,
`<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` are user-defined (type 4). ChatML template;
generation prompt ends `<think>\n` (or an empty think block with thinking off); earlier reasoning kept by default.

## 9. Newer versions
No CHANGELOG in this repo. Checkouts 0142, 0143, 0144, 01441, 0145 exist on disk (not read).

## Facts for attaching the computer
1. The cleanest injection point is `Service.run`'s continuation (`server.py:3621-3654`): STOP, re-GEN with prompt +
   generated + injected ids, reusing live state with no prefill, if the injected ids bypass the parser.
2. MCP rounds re-render via the template; reuse needs exact token match with the live sequence, else the turn
   checkpoint.
3. GDN state rolls back only via checkpoints.
4. Per-token state ~14 KB (KV + indexer) + ~118 MB fixed GDN.
5. Without vision there is room for a few-hundred-MiB module with no loss of resident experts; with vision it costs
   expert slots.

## Open questions that need a run (gateway, engine stopped first, one at a time)
- Exact VRAM after each startup step: `STRATA_TRACE=1` with the novision launch (`engine-cuda12/strata --serve <args>`),
  read "strata trace: ... MiB free after ..." lines.
- Memory plan without a GPU: `strata-plan` exists in source (`src/plan/plan_main.cpp`) but isn't built.
- After an MCP round, live reuse or checkpoint: one MCP request via the web app, then `RESUME`/`REUSED` and
  "prompt N tokens = X reused" in the log.
- Whether the 243 PAD rows are trained: read rows 248077+ of `token_embd`/`output` from shard 1 with numpy (CPU).
