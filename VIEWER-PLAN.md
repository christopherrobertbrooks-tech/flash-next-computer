# RISC-V viewer: watch the computer work (plan, 2026-10-30; nothing built yet)

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
   tokens (via `on_token`), so the right side shows the CPU state **as the circuit in the weights produces it**.
   Every token is checked against the shadow; a mismatch is flagged and logged (below), not a halt. Limited to the calculator
   program (the only firmware Qwenputer runs); ~1,700 steps/s, so it needs a speed control or step-on-demand.

## Honesty rules for the display
- Phases 1-2 show the **reference machine**, not the model. Phase 3 shows the reference machine's state **verified
  token by token** against the model's circuit. The label on screen says which.
- **Phase 3 runs the calculator program only.** It's the only firmware Qwenputer runs, so the live inside view shows
  calculations (e.g. 123456789 x 987654321 step by step), not arbitrary programs. Said on screen too.
- **A mismatch in phase 3 is flagged and logged, not a halt.** At the first token where the circuit and the shadow
  differ: the step is marked (both values shown), written to a log with the position and the recent tokens, and the
  status line changes from "verified" to "UNVERIFIED since step N" for the rest of that answer. The chat continues;
  the right side keeps showing the shadow, labelled as no longer verified. Why not a hard stop: a rare checker or
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
