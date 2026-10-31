"""RISC-V viewer, phase 3 (VIEWER-PLAN.md v2): watch Qwenputer's computer while the model answers.

Qwenputer-4B answers a question on the V100 through upstream's own generation loop (fast circuit executor). Every
token it produces is passed to a **shadow**: upstream's Python RV32 reference machine, stepped one record at a time.
A record's state is shown only after the model's circuit produced exactly that record's tokens, so everything on the
right side is a reference state the circuit has matched token by token. On the first token that differs, the shadow
stops: the last verified state stays on screen, the label turns to "UNVERIFIED since record N" for the rest of the
answer, and the first disagreement is logged. The chat continues.

The label is honest about what runs: the circuit is upstream's fp32 executor built from the circuit description; the
checkpoint's bf16 computer-layer modules are not executed on this path. Only the calculator program runs (calc-bridge.elf,
the one firmware Qwenputer carries).

    python rvlive.py MODEL_DIR "What is 123456789 * 987654321? Use the calc tool."    (on the gateway, V100 visible)
    add --headless to print the final screen and the verification summary instead of opening the terminal view

Keys: space = one record, r = run/pause, + / - = speed, q = quit (stops the model too). While paused, the model's own
text streams freely; the computer waits for you. The model itself pauses when the viewer is behind (bounded queue).
"""
import argparse, collections, json, queue, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rvview as V  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import elf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

FIRMWARE = Path(BR.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"
LABEL = ("SHADOW: reference machine checked token by token against Qwenputer's fast circuit executor "
         "(checkpoint bf16 computer layers not executed)")


class Shadow:
    """The reference machine in lockstep with the model's token stream (stream tokens: chip tokens as spec values,
    text as TEXT_TOKEN | id). feed() returns "text", "chip" or "committed" (a record's tokens all matched)."""

    def __init__(self, T, prompt_ids, log_path=None):
        self.T = T
        m = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(m)
        self.sess = BR.Session(T, m)
        self.sess.history(prompt_ids)
        self.m = m
        self.in_call = self.started = False
        self.expect = collections.deque()
        self.pending = None            # the record being matched: (before-state, its expected tokens)
        self.ends_call = False
        self.verified = True
        self.mismatch = None
        self.records = self.refreshes = self.tokens_checked = self.calls = 0
        self.text_ids = []
        self.log_path = log_path
        self.last = {"kind": "start", "pc": m.pc, "text": "", "regs": set(), "digits": set(), "ram": set()}
        # rvview's renderer reads these
        self.source, self.program, self.note, self.labels = LABEL, "calc-bridge.elf (the chat calculator)", "", {}
        self.words = 1 + max((i for i, w in enumerate(m.rom) if w), default=0)

    @property
    def left(self):
        text = self.T.decode_bytes(self.text_ids).decode(errors="replace")
        return ["Qwenputer's answer (text as it streams):", ""] + text.splitlines()[-200:]

    def status(self):
        if self.verified:
            return f"VERIFIED: {self.tokens_checked} computer tokens matched, {self.records} records, {self.calls} calls"
        return f"UNVERIFIED since record {self.mismatch['record']}: {self.mismatch['why']}"

    def _next_record(self):
        m = self.m
        pc, steps = m.pc, m.steps
        text = V.disasm(m.rom[(pc - S.ROM_BASE) >> 2], pc, m.x)
        before = (pc, steps, text, list(m.x), [list(r) for r in m.bn], list(m.ram))
        rec = m.record()
        exp = []
        for t in rec:
            exp.append(t)
            fam, p = S.family(t)
            if fam == "BYTE":
                q = self.T.printable[p] if p < len(self.T.printable) else self.T.NL
                exp.append(BR.text_token(q))
        self.ends_call = rec[-1] == S.tok("HALT")
        if self.ends_call:
            exp.append(BR.text_token(self.T.NL))     # the HALT echo the reference expects next
        self.pending = (before, rec)
        self.expect.extend(exp)

    def _commit(self):
        (pc, steps, text, x0, bn0, ram0), rec = self.pending
        m = self.m
        refresh = bool(rec) and S.family(rec[0])[0] == "R" and m.steps == steps
        self.records += 1
        self.refreshes += refresh
        self.last = {"kind": "refresh" if refresh else "instruction", "pc": pc,
                     "text": "(refresh record: the trace re-anchors the pc)" if refresh else text,
                     "regs": {k for k in range(32) if m.x[k] != x0[k]},
                     "digits": {k for k in range(8) if m.bn[k] != bn0[k]},
                     "ram": {k for k in range(len(m.ram)) if m.ram[k] != ram0[k]}}
        self.pending = None
        if self.ends_call:
            self.in_call = False
            if m.fault is None:
                m.halted = False                         # parked: the next GO resumes after the exit ecall

    def _flag(self, why, got, want):
        self.verified = False
        self.mismatch = {"record": self.records + 1, "why": why, "got": got, "want": want,
                         "recent_text": self.T.decode_bytes(self.text_ids[-40:]).decode(errors="replace")}
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(self.mismatch) + "\n")

    def feed(self, st):
        if S.is_text(st):
            self.text_ids.append(st - S.TEXT_TOKEN)
        if not self.verified:                            # stopped at the first disagreement: nothing more is fed
            return "text" if S.is_text(st) else "chip"
        if self.in_call:
            if not self.expect:
                if self.pending is None and not self.started:
                    self.expect.append(S.tok("GO"))      # the run starts with GO
                    self.started = True
                    self.m.halted = False
                else:
                    self._next_record()
            want = self.expect.popleft()
            if st != want:
                self._flag("the circuit's token differs from the reference's", S.name(st) if st < S.VOCAB else st,
                           S.name(want) if want < S.VOCAB else want)
                return "text" if S.is_text(st) else "chip"
            self.tokens_checked += 1
            if not self.expect and self.pending is not None:
                self._commit()
                return "committed"
            return "text" if S.is_text(st) else "chip"
        if not S.is_text(st):
            self._flag("a computer token outside a tool call", S.name(st), "text")
            return "chip"
        q = st - S.TEXT_TOKEN
        self.sess.qwen([q], run=False)                   # the call's capture, exactly as the reference fills it
        if q == self.T.TCC:
            self.in_call, self.started, self.calls = True, False, self.calls + 1
        return "text"


# ---- the model side: a worker thread, a bounded queue (the model pauses when the viewer is behind) ---------------
class Stopped(Exception):
    pass


def start_model(model_dir, question, max_new, max_chip):
    from chipformer.rv import chat as CH
    chat = CH.RvChat(model_dir, executor=True)
    ids = chat.prompt_ids([{"role": "user", "content": question}])
    q, stop = queue.Queue(maxsize=4096), threading.Event()

    def on_token(t):
        if stop.is_set():
            raise Stopped()
        q.put(t)

    def work():
        m = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(m)
        try:
            r = chat.generate(ids, m, max_new, max_chip=max_chip, on_token=on_token)
            q.put(("done", {k: r.get(k) for k in ("text_tokens", "chip_tokens", "seconds")}))
        except Stopped:
            q.put(("done", {"stopped": True}))
        except Exception as e:  # noqa: BLE001
            q.put(("done", {"error": f"{type(e).__name__}: {e}"}))

    threading.Thread(target=work, daemon=True).start()
    return chat, ids, q, stop


def to_stream(chat, t):
    return t - chat.v0 if t >= chat.v0 else BR.text_token(t)


def headless(args):
    chat, ids, q, stop = start_model(args.model, args.question, args.max_new, args.max_chip)
    sh = Shadow(chat.T, ids, args.log)
    while True:
        t = q.get()
        if isinstance(t, tuple):
            info = t[1]
            break
        sh.feed(to_stream(chat, t))
    print("\n".join(x.rstrip() for x in V.text_of(V.render(sh, args.rows, args.cols, 0), args.cols)))
    print(json.dumps({"status": sh.status(), "generate": info, "output": bytes(sh.m.out).decode(errors="replace")}))


def tui(args):
    import curses
    chat, ids, q, stop = start_model(args.model, args.question, args.max_new, args.max_chip)
    sh = Shadow(chat.T, ids, args.log)

    def main(scr):
        curses.curs_set(0)
        curses.use_default_colors()
        style = {"": curses.A_NORMAL, "dim": curses.A_DIM, "hl": curses.A_BOLD, "label": curses.A_BOLD,
                 "rev": curses.A_REVERSE}
        st = {"running": False, "speed": 3, "done": None, "step": False, "credit": 0.0, "t": time.time()}
        while True:
            # the model's own text flows freely; computer records are paced: one per space press, or "speed" per second
            now = time.time()
            if st["running"]:
                st["credit"] = min(st["credit"] + V.SPEEDS[st["speed"]] * (now - st["t"]), V.SPEEDS[st["speed"]])
            st["t"] = now
            t_end = now + 0.04
            while st["done"] is None and time.time() < t_end and not q.empty():
                if sh.in_call and sh.verified and not (st["step"] or st["credit"] >= 1):
                    break                                  # the computer waits for the controls
                t = q.get_nowait()
                if isinstance(t, tuple):
                    st["done"] = t[1]
                    break
                if sh.feed(to_stream(chat, t)) == "committed":
                    if st["step"]:
                        st["step"] = False
                    else:
                        st["credit"] -= 1
            rows, cols = scr.getmaxyx()
            sh.note = sh.status() + ("" if st["done"] is None else "   | answer finished")
            scr.erase()
            for y, row in enumerate(V.render(sh, rows, cols, V.SPEEDS[st["speed"]], st["running"])):
                x = 0
                for s_, txt in row:
                    if x >= cols:
                        break
                    try:
                        scr.addnstr(y, x, txt, cols - x, style.get(s_, curses.A_NORMAL))
                    except curses.error:
                        pass
                    x += len(txt)
            scr.refresh()
            scr.timeout(50)
            k = scr.getch()
            if k in (ord("q"), 27):
                stop.set()
                return
            if k == ord(" "):
                st["step"] = True
            elif k == ord("r"):
                st["running"] = not st["running"]
            elif k in (ord("+"), ord("=")):
                st["speed"] = min(len(V.SPEEDS) - 1, st["speed"] + 1)
            elif k == ord("-"):
                st["speed"] = max(0, st["speed"] - 1)

    curses.wrapper(main)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("question")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--max-chip", type=int, default=2_000_000)
    ap.add_argument("--log", default="rvlive-mismatches.jsonl")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--rows", type=int, default=40)
    args = ap.parse_args()
    (headless if args.headless else tui)(args)


if __name__ == "__main__":
    main()
