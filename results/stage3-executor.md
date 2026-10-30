# Stage 3, first measurement: the computer's executor on its own (2026-10-30)

`scripts/executor_footprint.py` (log `executor-footprint.log`): upstream's Triton Executor2 (fp32 circuit arithmetic),
built without the language model, on the V100 through the lock (1380 MHz), torch 2.14.1+cu126, triton 3.8.0,
program `c-crc32` (4,166 steps).

| Circuit | Dims | Executor allocation | Peak allocation (incl. check) | Exact vs reference | Steps/s (CUDA graph) |
|---|---|---|---|---|---|
| Bare RV32 core | 2,679 | 20.8 MiB | 63.4 MiB | all 4,166 | 2,157 |
| Qwenputer circuit (core + text bridge) | 2,918 | 21.2 MiB | 67.6 MiB | all 4,166 | 1,989 |

- **A separate process pays a CUDA context first:** a PyTorch process holding one tensor showed **368 MiB** in
  nvidia-smi on the V100 (the script's own "60 MiB context" line is wrong: its baseline was taken after the context
  already existed). So the computer as its own process costs about **440 MiB** of VRAM; inside an existing CUDA process
  (e.g. ported into Strata) about 20-70 MiB.
- **On the CPU** (fastexec, gateway 13700KF, 16 threads, timing includes the reference comparison): about 52 steps/s
  sparse, 27 dense. A calculation is a few thousand computer steps, so ~40-100 s per call: not practical.

**Against Strata's room (mapping):** vision off leaves 661 MiB free with all experts resident; vision and MTP off,
1,849 MiB. A ~440 MiB separate process fits the first only tightly (and Strata may need some of that free room while
running); it fits the second comfortably, at MTP's speed cost. Next measurement: Strata's real startup and running
memory with vision off, with and without the computer process already holding its memory (how many expert slots it
displaces).
