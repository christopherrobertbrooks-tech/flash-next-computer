# Stage 2 mapping: the deployed Strata (v0.1.41, fb58e0d) for Flash-Next Coder

Read-only map made 2026-10-30 by a research agent from `/mnt/steam/Strata-0141` source, the coder-iq1_m pack, the GGUF
header (metadata only) and existing logs. Nothing was launched or changed. Items marked UNCONFIRMED need a run.
Full report: `strata-0141-map.md`.

Integration-relevant in one place:
- Architecture `qwen4exp`: 48 layers = 36 Gated DeltaNet + 12 "QSA" full-attention (every 4th); hidden 2560 with
  4 hyper-connection streams (residual 10,240 floats); 256 of 512 experts, top-10, + 1 shared; PLE n-gram table on SSD.
  Vocabulary 248,320 with only 243 unused slots.
- No public HTTP raw-token endpoint (the engine's private stdin protocol takes token ids); the server's reasoning
  wrap-up continues generation with added text (server.py:3621-3654; those tokens are processed by the model like any
  others); server-side tools already exist (MCP, opt-in per request, re-render through the template).
- GDN state can't be rolled back except via checkpoints; speculative verify replays accepted tokens into GDN.
- Memory (Oct 18/23 logs): vision costs ~1.4 GiB; no vision = all 12,288 experts resident with 661 MiB free; no
  vision + no MTP = 1,849 MiB free. (Measured in stage 3: a 434 MiB companion process still displaces 136 slots.) Activations/residual fp32; GDN state fp32; KV int8 (~12.7 KB/token).
