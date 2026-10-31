# Review brief for Codex: inside-4b (in progress) and the viewer plan (two passes)

Branch `inside-4b` (from `main` bc9ecae). The full run started 7:46 pm under tag `inside4b-freeze` and is still going
(R first, then C); its results are **not** part of this review. Work in your own clone, branch
`review/codex-inside4b` from `inside-4b`. Review only: no GPU, no model runs, nothing pushed to `main` or `inside-4b`.
Gateway, read-only: `/mnt/nvme/flashnext-test/inside4b/` (pilot folders, scripts as deployed), upstream Qwenputer
`/mnt/nvme/qwenputer/qwenputer/experimental/riscv` (`chipformer/rv/chat.py`, `bridge.py`, `ref.py`, `env/calc/calc.c`),
the models `/mnt/nvme/qwenputer/model`, `/mnt/nvme/qwen35-4b/model` (config/index files only; don't load weights).

## Pass 1: blind (commit and push before opening `REVIEW-INSIDE4B-PASS2.md`)
**Read:** `INSIDE4B-PLAN.md`, `VIEWER-PLAN.md`, `inside4b/` (run_qwenputer.py, run_plain.py, final-primary.jsonl),
the changes to `stage5/score.py` since `main`, the pilot raw data `results/inside4b-pilot/` (gzipped replies).
**Don't read yet:** commit messages on `inside-4b`, or `REVIEW-INSIDE4B-PASS2.md`.

Answer:
1. **The comparison:** do arms C (Qwenputer, computer in the weights) and R (plain Qwen3.5-4B + upstream's Python
   reference machine running the same firmware) differ only in where the CPU lives? Same prompt ids, decoding,
   budgets, response tokens, expression capture? Anything unfair to either?
2. **The runners:** does `run_plain.py` reproduce the firmware's capture and response exactly (arm B's Python recipe
   and arm R's emulator)? Does `run_qwenputer.py` record mismatches (lock errors) as hard failures, and the computer's
   limits correctly? Can a wrong answer pass or a right one fail in scoring?
3. **The pilot:** rescore it independently. Are its claims sound (11/12 for B, R and C; R and B identical on 14/15;
   Qwenputer and plain 4B not text-identical; speed and memory)?
4. **The base-model claim:** the plan says Qwenputer's Qwen weights equal the base model's apart from a norm rescale
   of sqrt(2560/3968). Check it from the checkpoints (index metadata and a few tensors via safetensors slices, no full
   loads into GPU).
5. **The viewer plan:** feasible as written? Is the honesty framing right (reference machine vs the model's circuit;
   phase 3's lockstep shadow)? What's missing or risky?

Write `REVIEW-INSIDE4B-FINDINGS.md` ("Pass 1 (blind)"), commit and push.

## Pass 2: guided
Open `REVIEW-INSIDE4B-PASS2.md`, answer it as "Pass 2 (guided)" in the same file, push.
