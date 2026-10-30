# Stage 3, second measurement: Strata with the computer beside it (2026-10-30, 3:19-3:31 am)

`scripts/strata_mem_test.sh` (`scripts/computer_holder.py` for the computer): Strata 0.1.41 launched by itself on
port 8091 (not llama-swap; live config and logs untouched), `STRATA_TRACE=1`, V100 lock (1380 MHz). One request each
run: a 1,446-token coding prompt, 600 tokens out, temperature 0. Raw: `stage3-strata-runs/`.

| Run | Setup | Expert slots | Free after load | Lowest free while working | Request time | Mean |
|---|---|---|---|---|---|---|
| A x3 | vision off | 12,288 (all) | 660 MiB | 540 MiB | 9.3, 9.3, 9.6 s | **9.4 s** |
| C x3 | vision off + computer process (434 MiB, idle) | 12,152 | 488 MiB | 378 MiB | 10.7, 10.3, 10.3 s | **10.4 s** |
| D x3 | today's normal: vision on (strata-vision 1,430 MiB) | 11,642 | 486 MiB | n/a | 11.0, 10.3, 10.2 s | **10.5 s** |
| B x1 | vision off + computer computing non-stop | 12,152 | 488 MiB | 378 MiB | 23.3 s | worst case |

- The computer as its own process takes **434 MiB** (nvidia-smi), passed its exactness check (graph replays vs the
  reference) and ran 1,963 steps/s alone.
- Holding that memory costs Strata 136 expert slots (1.1%) and ~11% on this request; prompt reading is unchanged
  (~1.8 s in every run), so the cost is in decoding (most likely the missing experts running on the CPU).
- **Vision off + the computer is as fast as today's normal setup with vision on** (10.4 vs 10.5 s): the computer uses
  the room vision used, and more experts stay resident (12,152 vs 11,642).
- **Computing at the same time is the real cost:** in run B the request took 2.5x as long and the computer dropped
  from ~1,850 to ~930 steps/s while Strata decoded. A design should let them take turns (the computer runs while
  Flash-Next waits for a tool result), not overlap.
- Stable in every run; no out-of-memory errors.
- Limits: one prompt, 3 runs per setup (B once), 1380 MHz lock clocks (Strata normally runs at default clocks).
- Run B's first attempt didn't start the computer (a config-reading bug in `computer_holder.py`, fixed); the script now
  aborts if the computer isn't ready.
