# Review brief for Codex: the RISC-V viewer (phases 1-3) and the speed research (two passes)

Two branches from `main` (20b9eab): `viewer-live` (viewer phases 1 and 3, the phase-2 premise check, launcher
fixes) and `computer-speed` (branched from `viewer-live` before its last two launcher commits: the speed session and
the flash-linear-attention test). Work in your own clone on `review/codex-viewer-speed`, made from `computer-speed`
with `viewer-live` merged in. Review only: no pushes to `main`, `viewer-live` or `computer-speed`.

**This time, GPU runs are allowed for checking claims**, one job at a time inside the V100 lock
(`/mnt/nvme/humming-fast/tools/v100 <command>`), and never touching the 4070. The viewer's headless modes and the
probes are fine to rerun. Don't change files outside your own scratch folders (`/mnt/nvme/flashnext-test/viewer`,
`/speed` and `/mnt/nvme/qwenputer` are ours: read them, don't edit). The separate fla folder
`/mnt/nvme/qwenputer/fla-pkgs` may be used via PYTHONPATH only.

## Pass 1: blind (commit and push before opening `REVIEW-VIEWER-SPEED-PASS2.md`)
**Read:** `VIEWER-PLAN.md` (v2 and its status), `viewer/` (rvview.py, rvlive.py, test_rvview.py, test_rvlive.py,
asm_premise.py, the two launchers), `SPEED-PLAN.md`, `speed/` (speed_probe.py, fla_probe.py), the raw results in
`results/viewer/` and `results/speed/` (probe.json, llama-bench.json, session.log, fla-*.json).
**Don't read yet:** `results/speed-results.md`, commit messages on the two branches, or the pass-2 file.

Answer:
1. **Viewer honesty:** does what's on screen match what's true? Phase 1 shows the reference machine; phase 3 a shadow
   checked against Qwenputer's executor (the checkpoint's bf16 computer layers aren't run). Is "VERIFIED" ever shown for
   something that wasn't checked? Does the shadow stop correctly at a mismatch, and can it silently realign?
2. **Phase 3's lockstep:** does `Shadow.feed` reproduce `bridge.Session.run` exactly (GO, records, BYTE echoes,
   the HALT newline, parking between calls, capture through `qwen(run=False)`)? The pacing and the bounded queue
   (does the model really pause; can tokens be lost or reordered; what happens on quit)?
3. **The tests:** do `test_rvview.py` and `test_rvlive.py` actually fail when they should?
4. **The premise check:** is "the 4B can't write reliable RV32 assembly" fair from 16 attempts with that prompt?
5. **The speed numbers:** recompute from the raw files; are the methods sound (timed2, the round-trip loop, the
   prefill timing, the kernel profile, llama-bench vs the HF decode loops, the with/without-fla comparison)?
6. Anything wrong, overclaimed or missing.

Write `REVIEW-VIEWER-SPEED-FINDINGS.md` ("Pass 1 (blind)"), commit and push.

## Pass 2: guided
Open `REVIEW-VIEWER-SPEED-PASS2.md`, answer it in the same file as "Pass 2 (guided)", push.
