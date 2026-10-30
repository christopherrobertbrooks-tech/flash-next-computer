# Review brief for Codex: flash-next-computer, stages 2-3 (two passes)

Chris's plan is `PLAN.md` (v2). This reviews stage 2 (a read-only map of the deployed Strata) and stage 3 (the
Qwenputer computer measured beside Strata, and wrapped as a reference-checked service), before stage 4 (wiring it to
Flash-Next through a copy of Strata's server).

## Where to work
Your own clone of `christopherrobertbrooks-tech/flash-next-computer` (Dev-Console `~/Code/flash-next-computer-codex`,
or one of your own on the gateway), branch `review/codex-stage23` from `stage3`. Not `~/Code/flash-next-computer`.
On the gateway, read-only: the deployed Strata `/mnt/steam/Strata-0141` (v0.1.41, fb58e0d), upstream Qwenputer
`/mnt/nvme/qwenputer/qwenputer` (ca13310 + `patches/` from the qwenputer-v100 repo), the test folders
`/mnt/nvme/flashnext-test/*`. Don't start Strata or anything on a GPU; don't change files there.

## Pass 1: blind (commit and push it before opening `REVIEW-PASS2.md`)
**Read:** `PLAN.md`; the code (`scripts/`, `computer/`); the raw outputs (`results/executor-footprint.log`,
`results/stage3-strata-runs/*`, `results/computer-service-test.log`, `results/computer-service-results.json`); the
Strata source and logs on the gateway; upstream Qwenputer's code.
**Don't read yet:** `README.md`'s status, `results/*.md`, `mapping/`, commit messages after `Plan (v2)`, or
`REVIEW-PASS2.md`.

Then write your own answers:
1. **Strata, from its source:** where could an exact computer attach to Flash-Next served by this Strata (tool flow,
   hidden-token injection, state handling, speculative decoding), and what are the main risks?
2. **The measurements:** what do the raw run outputs show about memory, expert slots and speed with and without the
   computer process and vision? Are the methods sound? What's missing?
3. **The service (`computer/`):** is the per-token reference check sound and complete? What does the test corpus
   establish, and what doesn't it?
4. Anything wrong, fragile or overclaimed; what you'd want before stage 4.

Write `REVIEW-FINDINGS.md` ("Pass 1 (blind)"), commit and push.

## Pass 2: guided
Open `REVIEW-PASS2.md`, answer it as "Pass 2 (guided)" in the same file, push.

## Ground rules
Review only: no GPU, no downloads, no Strata launches, don't push to `main` or `stage3`.
