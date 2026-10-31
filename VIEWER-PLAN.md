# RISC-V viewer: watch the computer work (plan v2, 2026-10-30; nothing built yet)

v2 adds Codex's review (review/codex-inside4b 124755d / 5f0972b): when a step counts as verified, the
calculator's digit hardware, replay labels, and the live-view event contract.

## What Chris would have
A terminal app (dark, minimal) on the gateway, run from a terminal (`ssh -t ember-gateway ...`):
- **left:** the conversation with the model;
- **right:** the RV32 machine live: pc, the current instruction (disassembled), registers x0-x31 with ABI names
  (changed ones highlighted), a memory/stack window around sp, the program's output, the step count.

Keys: space = one step, r = run/pause, + / - = speed, q = quit.

## What exists already (upstream Qwenputer, on the gateway)
- `chipformer.rv.ref.Machine`: the Python RV32 reference machine. Registers `x[0..31]`, `pc`, `ram` words, `steps`,
  `out`; `record()` runs one instruction and returns that step's trace tokens: the same tokens Qwenputer's circuit
  produces, which is how every computer step has been checked so far.
- `chipformer.rv.asm.assemble(src)`: a two-pass RV32IM assembler (labels, sections, pseudo-instructions).
- `chipformer.rv.ref.decode(word)`: instruction decoding (op, rd, rs1, rs2, imm) for disassembly.
- `chipformer.rv.calc.machine(expr)`: the calculator program with an expression loaded.
- `RvChat.generate(..., on_token=...)`: Qwenputer's generation loop reports every token as it comes, chip (trace)
  tokens included.
- Libraries on the gateway: `curses` and `rich` (no installs needed).

## Phases
1. **Stepper (CPU only; can be built while the V100 is busy).** `viewer/rvview.py` with three sources:
   - `--asm FILE`: assemble a small program and step it (teaching mode, e.g. a function prologue moving sp);
   - `--calc EXPR`: step the calculator program on an expression: what the computer does inside a tool call;
   - `--replay RUN_ROW`: a saved Qwenputer answer: the chat text on the left; each tool call re-run on the reference
     machine on the right, labelled "replayed on the reference machine".
   Plus `--headless N` (step N times, print the state) so it can be tested without a screen.
2. **Live, beside:** the plain Qwen3.5-4B writes short RISC-V assembly when it wants to show something; the viewer
   assembles and steps it on the emulator as the reply streams. Needs a prompt or tool definition for "run this
   assembly" (the 4B was never trained on it: a premise check, it may write poor assembly).
3. **Live, inside (the striking one):** Qwenputer answers; a shadow reference machine runs in lockstep with its trace
   tokens (via `on_token`): the right side shows **a reference shadow checked against Qwenputer's fast circuit
   executor** (the checkpoint's bf16 computer layers aren't executed on that path). A mismatch is flagged and logged
   (below), not a halt. Limited to the calculator
   program (the only firmware Qwenputer runs); ~1,700 steps/s, so it needs a speed control or step-on-demand.

## Honesty rules for the display
- Phases 1-2 show the **reference machine**, not the model. Phase 3 shows the reference machine's state **verified
  token by token** against the model's circuit. The label on screen says which.
- **Phase 3 runs the calculator program only.** It's the only firmware Qwenputer runs, so the live inside view shows
  calculations (e.g. 123456789 x 987654321 step by step), not arbitrary programs. Said on screen too.
- **A mismatch in phase 3 is flagged and logged, not a halt. One behaviour (Codex re-check):** at the first token
  where the executor and the shadow differ, the last verified snapshot is kept in its own panel; the step is marked
  (both values shown) and written to a log with the position and the recent tokens; the status line changes from
  "verified" to "UNVERIFIED since step N" for the rest of that answer, and it can't turn back to verified until the
  next answer. The chat continues. The shadow stops at the mismatch (it isn't fed anything after it, so it can't
  silently realign); the main panel shows the frozen last verified state. This is a demonstration rule, not
  experiment verification. Why not a hard stop: a rare checker or
  precision artifact shouldn't kill the demo. There's one on record: the all-fp32 module path mispredicted 1 of 193
  tokens on a test program's prefill in qwenputer-v100 (cause not proven), while the executor path phase 3 uses had
  0 mismatches over 2.2 million computer tokens. Implementation note: upstream's `generate(lock=...)` breaks at the
  first lock error, so phase 3 runs without `lock` and keeps its own shadow machine fed from `on_token`, which can
  flag and continue.
- Values decoded straight from the model's trace tokens (`spec.decode_value` on value records) can be shown beside the
  shadow's, as a check, in phase 3.

## Open questions
- Running a TUI over RDP/ssh from Dev-Console: terminal size, colours (dark, near-colourless).
- Phase 3 needs the V100 (Qwenputer ~20 GB), so it can't run while experiments use the card.
- Phase 2: does the 4B write correct assembly at all? Test before building UI for it.

## Order
Phase 1 now (CPU). Phase 3 after the inside-4b run (it needs the card). Phase 2 only if its premise check passes.

## v2 additions (Codex)
- **Show the digit file, not just x0-x31.** The calculator does its arithmetic with custom digit-register ops
  (BMAC, BMSB, BLD, BGET, BSCAN, BPRD on `Machine.bn`); a plain RV32IM view would hide most of the work.
- **A step is one record, not one token.** `Machine.record()` commits a whole instruction (or a refresh record) and
  returns several trace tokens. The display advances only when a complete record has been checked.
- **Phase 3: verify before showing.** Upstream's `on_token` fires before the token is checked, so the viewer buffers
  each record's tokens, checks them against the shadow, and only then shows the committed state. On a mismatch, the
  rule above.
- **Phase 3 needs a real event design:** a bounded queue between the generating model (GPU) and the screen, with
  backpressure; a stated choice between pausing the model and pausing only the playback; cancellation; budget,
  partial-record, end-of-answer and resize states.
- **Labels on screen, always visible:** the source ("reference machine", "replayed from saved call text: original
  computer trace not recorded", or "shadow verified against Qwenputer's executor"), the program (`calc.elf` for
  `--calc` vs `calc-bridge.elf` in chat), and the path (executor vs bf16 computer-layer modules).
- **Record for reproducible replays:** future runs should save the original model and trace token ids, firmware and
  tokenizer hashes, the runtime path, prompt ids, verification status and why the answer ended. Today's saved replies
  have only text and counters, so phase-1 replay can only re-run the saved expression on the reference machine.
- **Headless tests:** branches, loads/stores, ABI names, refresh records, the custom digit ops, a mismatch, end of
  program, terminal resize.

## Status (2026-10-31)
- **Phase 1 built:** `viewer/rvview.py` (`--program`, `--asm`, `--calc`, `--replay`, `--headless`); controls in
  `viewer/test_rvview.py`, all passing on the gateway CPU.
- **Phase 3 built and run live:** `viewer/rvlive.py`. Qwenputer answered 123456789 x 987654321 on the V100 with the
  shadow VERIFIED on all 2,460 checked stream tokens (2,430 computer tokens and 30 text echoes; 1,570 records, 0
  mismatches); CPU controls in `viewer/test_rvlive.py` (one call, two calls, a single wrong token caught at its exact
  position, a computer token outside a call).
- **Fixed after Codex's review (review/codex-viewer-speed 809ed9c / 2970e32):** the screen now shows a separate
  committed snapshot, updated only after a record and all its echoes matched (before, it showed the checker machine,
  one record ahead of the check); an independent witness machine checks this at every token, and a control shows the
  old design is caught. The answer's end is explicit: only an EOS outside a call is "VERIFIED, answer complete";
  an end inside a call, a budget stop, a quit or a model error is INCOMPLETE (headless exits 1). Quit stops the model
  even when the queue is full, and waits for it. Pausing pauses playback only: the model keeps generating until the
  4,096-token queue is full. Labels: a short source line ("SHADOW vs fp32 executor; bf16 layers NOT run"),
  BMAC-overwrite shown as "=", and the replay label says its capture is approximate.
  Codex's re-check (1eb2056 / 82a4484) closed the snapshot, queue and label findings and found one more gap: an end
  while the model was still writing a call (after `<tool_call>`, before `</tool_call>`) still showed VERIFIED. Fixed:
  that counts as inside the call too (a sweep over every prefix of a two-call stream; the old rule is caught on 60).
  Headless always stops the model, even if the viewer itself fails; quitting the terminal view before the end shows
  INCOMPLETE (stopped). Live controls: 38, all pass; Codex confirmed the fix (7d4f7a8).
  Launch: `ssh -t ember-gateway /mnt/nvme/flashnext-test/viewer/rvlive.sh` (default question), or with your own question inside outer single quotes: `ssh -t ember-gateway '/mnt/nvme/flashnext-test/viewer/rvlive.sh "your question"'`.
- **Phase 2's premise failed:** `viewer/asm_premise.py` (`results/viewer/asm-premise.jsonl`). Plain Qwen3.5-4B wrote a
  correct, running RV32IM program for **0 of 8** small tasks without thinking and **3 of 8** with thinking (factorial,
  the function with a prologue, Fibonacci). The failures were logic bugs in plausible code (an off-by-one sum of 45,
  the wrong register returned, a max that keeps the last value) and degenerate repetition with thinking (the same line
  until the token budget ran out). So "the model writes programs to demonstrate" is unreliable for unattended demos under this tested setup (greedy,
  these prompts, this 4B model). The
  honest substitute: the model explains, and the viewer steps one of a library of **verified** example programs
  (phase 1's `--program` already does that).
