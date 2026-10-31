"""RISC-V viewer, phase 1 (VIEWER-PLAN.md v2): step the RV32 reference machine and watch its state.

Runs on the gateway's CPU (upstream Qwenputer's `chipformer.rv` on PYTHONPATH); no model, no GPU.

    python rvview.py --program call                 a built-in demo (call: recursive factorial and sum, stack frames)
    python rvview.py --asm FILE.s                   your own RV32IM assembly (upstream's assembler)
    python rvview.py --calc "123456789*987654321"   the calculator program on an expression
    python rvview.py --replay REPLIES.jsonl[.gz] ID a saved Qwenputer answer: its text on the left, each tool call
                                                    re-run on the reference machine on the right
    add --headless N to step N records (or "end") and print the state instead of opening the screen

Keys: space = one step, r = run/pause, + / - = speed, n = next tool call (replay), q = quit.

What you're watching is always upstream's **reference machine** (the Python RV32 emulator that checked every step of
Qwenputer's computer in our tests), never the model itself; the label at the top of the right side says so. One step is
one record: one whole instruction, or a "refresh" record the trace format inserts every 63 straight-line instructions.
The calculator does its arithmetic in a decimal digit file (8 registers of 128 digits, custom BMAC/BMSB/BLD/BGET/BPRD/
BSCAN instructions), shown under the ordinary registers.
"""
import argparse, gzip, json, sys, textwrap
from pathlib import Path

from chipformer.rv import asm, calc, programs, ref
from chipformer.rv import spec as S

ABI = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1"] + [f"a{i}" for i in range(8)] \
    + [f"s{i}" for i in range(2, 12)] + ["t3", "t4", "t5", "t6"]
BRANCH = {"BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU"}
LOAD = {"LB", "LH", "LW", "LBU", "LHU"}
STORE = {"SB", "SH", "SW"}
I_ALU = {"ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI"}
SHIFT_I = {"SLLI", "SRLI", "SRAI"}
DIGIT_ROW = {"BMAC", "BMSB", "BLD"}
SPEEDS = [1, 2, 5, 10, 25, 100, 500, 2000]

REFERENCE = "REFERENCE MACHINE (Python RV32 emulator), not the model"
REPLAY = ("REPLAY: saved call text re-run on the reference machine (calc.elf); "
          "the original computer trace was not recorded")


def reg(i):
    return ABI[i]


def disasm(word, pc, x=None):
    """One instruction as text, with ABI register names. x (registers) adds the digit ops' decoded operands."""
    i = ref.decode(word)
    op = i.op
    lo = op.lower()
    if op in BRANCH:
        return f"{lo} {reg(i.rs1)}, {reg(i.rs2)}, {pc + i.imm:#x}"
    if op == "JAL":
        return f"jal {reg(i.rd)}, {pc + i.imm:#x}"
    if op == "JALR":
        return f"jalr {reg(i.rd)}, {i.imm}({reg(i.rs1)})"
    if op in LOAD:
        return f"{lo} {reg(i.rd)}, {i.imm}({reg(i.rs1)})"
    if op in STORE:
        return f"{lo} {reg(i.rs2)}, {i.imm}({reg(i.rs1)})"
    if op in ("LUI", "AUIPC"):
        return f"{lo} {reg(i.rd)}, {(i.imm >> 12) & 0xFFFFF:#x}"
    if op in I_ALU:
        return f"{lo} {reg(i.rd)}, {reg(i.rs1)}, {i.imm}"
    if op in SHIFT_I:
        return f"{lo} {reg(i.rd)}, {reg(i.rs1)}, {i.imm & 31}"
    if op == "ECALL":
        if x is not None:
            return "ecall  (" + {1: "putchar a0", 93: "exit a0"}.get(x[17], f"a7={x[17]}") + ")"
        return "ecall"
    if op == "BPRD":
        return f"bprd   print digit d{i.dd}[{i.imm}]"
    if op == "BSCAN":
        return f"bscan {reg(i.rd)}, {reg(i.rs1)}   (count decimal digits at {reg(i.rs1)})"
    if op in DIGIT_ROW or op == "BGET":
        s = f"{lo} {reg(i.rd)}, {reg(i.rs1)}, {reg(i.rs2)}"
        if x is not None:
            b = x[i.rs2]
            dd, ds, n, off, m = b & 7, b >> 3 & 7, b >> 7 & 63, b >> 13 & 127, min(b >> 20 & 15, 9)
            s += {"BMAC": f"   d{dd}[{off}..] += d{ds} x {m} ({n} digits)",
                  "BMSB": f"   d{dd}[{off}..] -= d{ds} x {m} ({n} digits)",
                  "BLD": f"   d{dd}[{off}..] <- {n} digits from memory",
                  "BGET": f"   {reg(i.rd)} <- d{dd}[{off}]"}[op]
        return s
    if op in ("ILLEGAL", "EBREAK", "FENCE", "MISA"):
        return lo
    return f"{lo} {reg(i.rd)}, {reg(i.rs1)}, {reg(i.rs2)}"


def digits(row):
    """A digit register (least significant digit first) as a number string."""
    s = "".join(str(d) for d in reversed(row)).lstrip("0")
    return s or "0"


class Session:
    """One machine being stepped, with what changed in the last step."""

    def __init__(self, machine, source, program, left, note="", labels=None):
        self.m, self.source, self.program, self.left, self.note = machine, source, program, left, note
        self.labels = {a: n for n, a in (labels or {}).items()}       # address -> label, for the code window
        self.words = 1 + max((i for i, w in enumerate(machine.rom) if w), default=0)
        self.records = self.refreshes = 0
        self.last = {"kind": "start", "pc": machine.pc, "text": "", "regs": set(), "digits": set(), "ram": set()}

    def step(self):
        m = self.m
        if m.halted:
            return False
        pc, steps = m.pc, m.steps
        word = m.rom[(pc - S.ROM_BASE) >> 2]
        text = disasm(word, pc, m.x)
        x0, bn0, ram0 = list(m.x), [list(r) for r in m.bn], list(m.ram)
        toks = m.record()
        self.records += 1
        refresh = bool(toks) and S.family(toks[0])[0] == "R" and m.steps == steps
        self.refreshes += refresh
        self.last = {"kind": "refresh" if refresh else "instruction", "pc": pc,
                     "text": "(refresh record: the trace re-anchors the pc)" if refresh else text,
                     "regs": {k for k in range(32) if m.x[k] != x0[k]},
                     "digits": {k for k in range(8) if m.bn[k] != bn0[k]},
                     "ram": {k for k in range(len(m.ram)) if m.ram[k] != ram0[k]}, "tokens": len(toks)}
        return True

    def run(self, n):
        done = 0
        while (n is None or done < n) and self.step():
            done += 1
        return done


# ---- sources ---------------------------------------------------------------------------------------------------
def from_program(name):
    src = programs.ALL[name]
    prog = asm.assemble(src)
    return Session(ref.Machine.from_program(prog), REFERENCE, f"built-in program '{name}' (RV32IM)",
                   ["Program source:"] + src.strip("\n").splitlines(), labels=prog.labels)


def from_asm(path):
    src = Path(path).read_text()
    prog = asm.assemble(src)
    return Session(ref.Machine.from_program(prog), REFERENCE, f"your program {Path(path).name}",
                   ["Program source:"] + src.strip("\n").splitlines(), labels=prog.labels)


def from_calc(expr):
    return Session(calc.machine(expr), REFERENCE,
                   "calc.elf: the calculator alone (chat tool calls use calc-bridge.elf)",
                   ["Expression:", "  " + expr, "", "The calculator reads it from RAM at 0x100, evaluates it",
                    "exactly in the digit file, prints the value and exits."], note=expr)


def load_row(path, qid):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] == qid:
                return r
    raise KeyError(f"{qid} not in {path}")


def replay_calls(row):
    """The saved reply text and its tool calls' expressions (as the firmware would capture them: newlines dropped)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inside4b"))
    import frames
    text = row["raw"]["choices"][0]["message"].get("full_text") or row["raw"]["choices"][0]["message"]["content"]
    calls = frames.parse(text)["calls"]
    exprs = [(c["expression"] or "").replace("\n", "") for c in calls if c["response"] is not None]
    saved = [c["response"] for c in calls if c["response"] is not None]
    return text, exprs, saved


def from_replay(path, qid, k=0):
    row = load_row(path, qid)
    text, exprs, saved = replay_calls(row)
    if not exprs:
        raise SystemExit(f"{qid}: the saved reply has no completed tool calls to replay")
    left = [f"Saved answer {qid} (arm {row.get('arm')}), call {k + 1} of {len(exprs)}:", ""] + text.splitlines()
    s = Session(calc.machine(exprs[k]), REPLAY, f"calc.elf on call {k + 1}: {exprs[k][:60]}", left,
                note=f"saved response: {saved[k]}")
    s.replay = (path, qid, k, len(exprs), saved[k])
    return s


# ---- the screen, as text (the same function draws the terminal and the headless tests) ------------------------
def wrap(lines, width):
    out = []
    for line in lines:
        out += textwrap.wrap(line, width, replace_whitespace=False, drop_whitespace=False) or [""]
    return out


def right_pane(s, width):
    m, L = s.m, s.last
    lines = [("label", s.source[:width]), ("label", s.program[:width])]
    if hasattr(s, "status"):                              # the live view: verification status, always near the top
        lines.append(("rev" if not s.verified else "hl", s.status()[:width]))
    lines.append(("", ""))
    status = ("HALTED, exit code %s" % m.exit_code if m.halted and m.fault is None else
              f"FAULT {S.FAULTS[m.fault[0]]} at {m.fault[1]:#x}" if m.fault else "running")
    lines += [("", f"pc {m.pc:#07x}   steps {m.steps}   records {s.records} ({s.refreshes} refresh)   {status}"[:width]),
              ("hl", f"last: {L['kind']} at {L['pc']:#07x}: {L['text']}"[:width])]
    if not m.halted:
        nxt = m.rom[(m.pc - S.ROM_BASE) >> 2]
        lines.append(("", f"next: {disasm(nxt, m.pc, m.x)}"[:width]))
    lines.append(("", ""))
    ncol = 4 if width >= 56 else 2                      # each cell needs 13 characters plus a gap
    col = width // ncol
    for r in range(32 // ncol):
        cells = []
        for c in range(ncol):
            k = c * (32 // ncol) + r
            cells.append((k in L["regs"], f"{ABI[k]:>4} {m.x[k]:08x}".ljust(col)[:col]))
        lines.append(("regs", cells))
    lines.append(("", ""))
    lines.append(("", "digit file (most significant first):"[:width]))
    for k in range(8):
        d = digits(m.bn[k])
        txt = f"d{k} {d}" if len(d) <= width - 4 else f"d{k} ...{d[-(width - 8):]}"
        lines.append(("hl" if k in L["digits"] else "", txt[:width]))
    lines.append(("", ""))
    sp = m.x[2]
    lines.append(("", f"stack (sp = {sp:#x}):"[:width]))
    base = (sp - S.RAM_BASE) >> 2
    for w in range(base, min(base + 6, len(m.ram))):
        if 0 <= w < len(m.ram):
            lines.append(("hl" if w in L["ram"] else "", f"  {S.RAM_BASE + 4 * w:#07x}: {m.ram[w]:08x}"[:width]))
    changed = sorted(L["ram"])
    if changed and not all(base <= w < base + 6 for w in changed):
        lines.append(("hl", f"  memory written: {', '.join(f'{S.RAM_BASE + 4 * w:#x}' for w in changed[:4])}"[:width]))
    lines.append(("", ""))
    lines.append(("", f"output: {m.out.decode(errors='replace')!r}"[:width]))
    if s.note:
        lines.append(("", s.note[:width]))
    return lines


def code_window(s, width, before=4, after=5):
    """The instructions around the pc, the current one marked, with labels."""
    m = s.m
    i0 = (m.pc - S.ROM_BASE) >> 2
    out = ["Code around the pc:"]
    for i in range(max(0, i0 - before), min(s.words, i0 + after + 1)):
        a = S.ROM_BASE + 4 * i
        mark = ">" if i == i0 and not m.halted else " "
        lab = (s.labels.get(a, "") + ":") if a in s.labels else ""
        out.append(f"{mark} {a:#07x} {lab:<8} {disasm(m.rom[i], a)}"[:width])
    return out + [""]


def render(s, rows, cols, speed=None, running=False):
    """The whole screen as (rows) lists of (style, text) runs, never wider than cols."""
    lw = max(20, int(cols * 0.42))
    rw = max(10, cols - lw - 3)
    left = code_window(s, lw) + wrap(s.left, lw)
    right = right_pane(s, rw)
    help_ = f" space step | r {'pause' if running else 'run'} | +/- speed ({speed}/s) | n next call | q quit "
    out = []
    for y in range(rows - 1):
        lt = left[y] if y < len(left) else ""
        lst = "hl" if lt.startswith(">") else "label" if y == 0 else "dim"
        row = [(lst, lt[:lw].ljust(lw)), ("dim", " | ")]
        if y < len(right):
            style, txt = right[y]
            if style == "regs":
                for changed, cell in txt:
                    row.append(("rev" if changed else "", cell))
            else:
                row.append((style, txt))
        out.append(row)
    out.append([("rev", help_[:cols].ljust(cols))])
    return out


def text_of(screen, cols):
    return ["".join(t for _, t in row)[:cols] for row in screen]


# ---- the terminal ----------------------------------------------------------------------------------------------
def tui(s, args):
    import curses

    def main(scr):
        curses.curs_set(0)
        curses.use_default_colors()
        style = {"": curses.A_NORMAL, "dim": curses.A_DIM, "hl": curses.A_BOLD, "label": curses.A_BOLD,
                 "rev": curses.A_REVERSE}
        state = {"s": s, "running": False, "speed": 2}
        while True:
            cur = state["s"]
            rows, cols = scr.getmaxyx()
            scr.erase()
            for y, row in enumerate(render(cur, rows, cols, SPEEDS[state["speed"]], state["running"])):
                x = 0
                for st, txt in row:
                    if x >= cols:
                        break
                    try:
                        scr.addnstr(y, x, txt, cols - x, style.get(st, curses.A_NORMAL))
                    except curses.error:
                        pass
                    x += len(txt)
            scr.refresh()
            sp = SPEEDS[state["speed"]]
            scr.timeout(max(1, int(1000 / sp)) if state["running"] else -1)
            k = scr.getch()
            if k == -1 and state["running"]:
                if not cur.step():
                    state["running"] = False
                continue
            if k in (ord("q"), 27):
                return
            if k == ord(" "):
                cur.step()
            elif k == ord("r"):
                state["running"] = not state["running"]
            elif k in (ord("+"), ord("=")):
                state["speed"] = min(len(SPEEDS) - 1, state["speed"] + 1)
            elif k == ord("-"):
                state["speed"] = max(0, state["speed"] - 1)
            elif k == ord("n") and hasattr(cur, "replay"):
                path, qid, i, n, _ = cur.replay
                state["s"] = from_replay(path, qid, (i + 1) % n)
                state["running"] = False
            elif k == curses.KEY_RESIZE:
                pass

    curses.wrapper(main)


def build(args):
    if args.program:
        return from_program(args.program)
    if args.asm:
        return from_asm(args.asm)
    if args.calc:
        return from_calc(args.calc)
    return from_replay(args.replay[0], args.replay[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--program", choices=sorted(programs.ALL))
    g.add_argument("--asm")
    g.add_argument("--calc")
    g.add_argument("--replay", nargs=2, metavar=("REPLIES", "ID"))
    ap.add_argument("--headless", help="step N records (or 'end') and print the screen as text")
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--rows", type=int, default=40)
    args = ap.parse_args()
    s = build(args)
    if args.headless:
        s.run(None if args.headless == "end" else int(args.headless))
        print("\n".join(t.rstrip() for t in text_of(render(s, args.rows, args.cols, 2), args.cols)))
        return
    tui(s, args)


if __name__ == "__main__":
    main()
