"""RISC-V viewer, phase 3 (VIEWER-PLAN.md v2): watch Qwenputer's computer while the model answers.

Qwenputer-4B answers a question on the V100 through upstream's own generation loop (its fast fp32 circuit executor;
the checkpoint's bf16 computer-layer modules are not run on this path). Every token it produces goes to a **shadow**:
upstream's Python RV32 reference machine, stepped one record at a time on a private checker machine. The screen shows a
separate **committed snapshot**, updated only after the circuit produced every token of a record (its BYTE echoes and
the HALT newline included) exactly as the reference does. So nothing on the right side is ever ahead of what was
checked. On the first token that differs, checking stops: the snapshot keeps the last verified state, the status turns
"UNVERIFIED since record N" for the rest of the answer, and the disagreement is logged. When the answer ends, the
shadow is finished explicitly: an answer that stopped inside a call or a record is INCOMPLETE, never VERIFIED.
Only the calculator program runs (calc-bridge.elf, the one firmware Qwenputer carries).

    python rvlive.py MODEL_DIR "What is 123456789 * 987654321? Use the calc tool."    (on the gateway, V100 visible)
    add --headless to print the final screen and the summary instead of opening the terminal view (exit 1 unless the
    answer completed VERIFIED)

Keys: space = one record, r = run/pause, + / - = speed, q = quit (stops the model). Pausing pauses playback; the model
keeps generating until the bounded queue (4,096 tokens) is full, then waits.
"""
import argparse, collections, json, queue, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rvview as V  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import elf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

FIRMWARE = Path(BR.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"
SOURCE = "SHADOW vs fp32 executor"
PROGRAM = "bf16 layers NOT run; calc-bridge.elf"


class Snap:
    """A committed copy of the machine state the screen reads (the ROM is shared: it never changes)."""

    def __init__(self, m, parked=False):
        self.rom, self.x, self.ram = m.rom, list(m.x), list(m.ram)
        self.bn, self.out = [list(r) for r in m.bn], bytes(m.out)
        self.anchor, self.n, self.steps = m.anchor, m.n, m.steps
        self.halted, self.fault, self.exit_code, self.parked = m.halted, m.fault, m.exit_code, parked

    @property
    def pc(self):
        return S.ROM_BASE + 4 * (self.anchor + self.n)


class Shadow:
    """The reference machine in lockstep with the model's token stream (stream tokens: chip tokens as spec values,
    text as TEXT_TOKEN | id). feed() returns "text", "chip" or "committed" (a record's tokens all matched)."""

    def __init__(self, T, prompt_ids, log_path=None):
        self.T = T
        cm = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(cm)
        self.sess = BR.Session(T, cm)
        self.sess.history(prompt_ids)
        self.cm = cm                                     # the private checker machine
        self.m = Snap(cm, parked=True)                   # what the screen shows: committed state only
        self.in_call = self.started = False
        self.expect = collections.deque()
        self.pending = None
        self.ends_call = False
        self.verified = True
        self.mismatch = None
        self.final = None                                # set by finish(): why the answer ended
        self.records = self.refreshes = self.stream_checked = self.chip_checked = self.calls = 0
        self.text_ids = []
        self.log_path = log_path
        self.last = {"kind": "start", "pc": cm.pc, "text": "", "regs": set(), "digits": set(), "ram": set()}
        self.source, self.program, self.note, self.labels = SOURCE, PROGRAM, "", {}
        self.words = 1 + max((i for i, w in enumerate(cm.rom) if w), default=0)

    @property
    def left(self):
        text = self.T.decode_bytes(self.text_ids).decode(errors="replace")
        return ["Qwenputer's answer (text as it streams):", ""] + text.splitlines()[-200:]

    @property
    def unfinished(self):
        return self.in_call or self.pending is not None or bool(self.expect)

    def status(self):
        counts = f"{self.stream_checked} stream tokens checked ({self.chip_checked} computer), {self.records} records"
        if not self.verified:
            return f"UNVERIFIED since record {self.mismatch['record']}: {self.mismatch['why']}"
        if self.final is None:
            return "VERIFIED so far: " + counts
        if self.unfinished:
            return f"INCOMPLETE ({self.final}, stopped inside a call); verified prefix: " + counts
        if self.final != "eos":
            return f"INCOMPLETE ({self.final}); verified prefix: " + counts
        return "VERIFIED, answer complete: " + counts

    def finish(self, reason):
        """The answer ended (reason: eos, text budget, computer budget, stopped, error ...)."""
        self.final = reason

    def _next_record(self):
        cm = self.cm
        pc, steps = cm.pc, cm.steps
        text = V.disasm(cm.rom[(pc - S.ROM_BASE) >> 2], pc, cm.x)
        rec = cm.record()                                # advances the private checker only
        exp = []
        for t in rec:
            exp.append(t)
            fam, p = S.family(t)
            if fam == "BYTE":
                q = self.T.printable[p] if p < len(self.T.printable) else self.T.NL
                exp.append(BR.text_token(q))
        self.ends_call = rec[-1] == S.tok("HALT")
        if self.ends_call:
            exp.append(BR.text_token(self.T.NL))         # the HALT echo the reference expects next
        self.pending = (pc, steps, text, rec)
        self.expect.extend(exp)

    def _commit(self):
        pc, steps, text, rec = self.pending
        cm, old = self.cm, self.m
        refresh = bool(rec) and S.family(rec[0])[0] == "R" and cm.steps == steps
        if self.ends_call:
            self.in_call = False
            if cm.fault is None:
                cm.halted = False                        # parked: the next GO resumes after the exit ecall
        new = Snap(cm, parked=self.ends_call)
        self.records += 1
        self.refreshes += refresh
        self.last = {"kind": "refresh" if refresh else "instruction", "pc": pc,
                     "text": "(refresh record: the trace re-anchors the pc)" if refresh else text,
                     "regs": {k for k in range(32) if new.x[k] != old.x[k]},
                     "digits": {k for k in range(8) if new.bn[k] != old.bn[k]},
                     "ram": {k for k in range(len(new.ram)) if new.ram[k] != old.ram[k]}}
        self.m = new                                     # the screen moves only now
        self.pending = None

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
        if not self.verified:                            # stopped at the first disagreement: nothing more is checked
            return "text" if S.is_text(st) else "chip"
        if self.in_call:
            if not self.expect:
                if self.pending is None and not self.started:
                    self.expect.append(S.tok("GO"))      # the run starts with GO
                    self.started = True
                    self.cm.halted = False
                else:
                    self._next_record()
            want = self.expect.popleft()
            if st != want:
                self._flag("the circuit's token differs from the reference's", S.name(st) if st < S.VOCAB else st,
                           S.name(want) if want < S.VOCAB else want)
                return "text" if S.is_text(st) else "chip"
            self.stream_checked += 1
            self.chip_checked += not S.is_text(st)
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


# ---- the model side: a worker thread, a bounded queue, cooperative stop ---------------------------------------
class Stopped(Exception):
    pass


def make_producer(generate, q_size=4096):
    """Runs generate(on_token) on a thread. Tokens go through a bounded queue; every put waits at most 0.1 s at a
    time and checks the stop flag, so a stop is honoured even when the queue is full. The last item is
    ("done", info) with info["reason"]. Returns (queue, stop event, thread)."""
    q, stop = queue.Queue(maxsize=q_size), threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def on_token(t):
        if stop.is_set() or not put(t):
            raise Stopped()

    def work():
        info = {}
        try:
            info = generate(on_token) or {}
        except Stopped:
            info = {"reason": "stopped"}
        except Exception as e:  # noqa: BLE001
            info = {"reason": "error", "error": f"{type(e).__name__}: {e}"}
        finally:
            put(("done", info))

    th = threading.Thread(target=work, daemon=True)
    th.start()
    return q, stop, th


def stop_producer(q, stop, th, timeout=15.0):
    """Ask the producer to stop and wait for it (draining the queue so a blocked put returns). True if it ended."""
    stop.set()
    t = time.time() + timeout
    while th.is_alive() and time.time() < t:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        th.join(timeout=0.1)
    return not th.is_alive()


def start_model(model_dir, question, max_new, max_chip):
    from chipformer.rv import chat as CH
    chat = CH.RvChat(model_dir, executor=True)
    ids = chat.prompt_ids([{"role": "user", "content": question}])

    def generate(on_token):
        m = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(m)
        r = chat.generate(ids, m, max_new, max_chip=max_chip, on_token=on_token)
        toks = r["tokens"]
        reason = ("eos" if toks and toks[-1] in chat.eos else
                  "computer budget" if r.get("chip_tokens", 0) >= max_chip else
                  "text budget" if r.get("text_tokens", 0) >= max_new else "ended early")
        return {"reason": reason, **{k: r.get(k) for k in ("text_tokens", "chip_tokens", "seconds")}}

    q, stop, th = make_producer(generate)
    return chat, ids, q, stop, th


def to_stream(chat, t):
    return t - chat.v0 if t >= chat.v0 else BR.text_token(t)


def headless(args):
    chat, ids, q, stop, th = start_model(args.model, args.question, args.max_new, args.max_chip)
    sh = Shadow(chat.T, ids, args.log)
    while True:
        t = q.get()
        if isinstance(t, tuple):
            info = t[1]
            break
        sh.feed(to_stream(chat, t))
    sh.finish(info.get("reason", "unknown"))
    print("\n".join(x.rstrip() for x in V.text_of(V.render(sh, args.rows, args.cols, 0), args.cols)))
    print(json.dumps({"status": sh.status(), "generate": info, "output": sh.m.out.decode(errors="replace")}))
    th.join(timeout=5)
    sys.exit(0 if sh.status().startswith("VERIFIED, answer complete") else 1)


def tui(args):
    import curses
    chat, ids, q, stop, th = start_model(args.model, args.question, args.max_new, args.max_chip)
    sh = Shadow(chat.T, ids, args.log)

    def main(scr):
        curses.curs_set(0)
        curses.use_default_colors()
        style = {"": curses.A_NORMAL, "dim": curses.A_DIM, "hl": curses.A_BOLD, "label": curses.A_BOLD,
                 "rev": curses.A_REVERSE}
        st = {"running": False, "speed": 3, "done": None, "step": False, "credit": 0.0, "t": time.time()}
        while True:
            now = time.time()
            if st["running"]:
                st["credit"] = min(st["credit"] + V.SPEEDS[st["speed"]] * (now - st["t"]), V.SPEEDS[st["speed"]])
            st["t"] = now
            t_end = now + 0.04
            while st["done"] is None and time.time() < t_end and not q.empty():
                if sh.in_call and sh.verified and not (st["step"] or (st["running"] and st["credit"] >= 1)):
                    break                                  # the computer waits for the controls
                t = q.get_nowait()
                if isinstance(t, tuple):
                    st["done"] = t[1]
                    sh.finish(t[1].get("reason", "unknown"))
                    break
                if sh.feed(to_stream(chat, t)) == "committed":
                    if st["step"]:
                        st["step"] = False
                    else:
                        st["credit"] -= 1
            rows, cols = scr.getmaxyx()
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
                return
            if k == ord(" "):
                st["step"] = True
            elif k == ord("r"):
                st["running"] = not st["running"]
                st["credit"] = 0.0                        # pausing doesn't leave spendable credit behind
            elif k in (ord("+"), ord("=")):
                st["speed"] = min(len(V.SPEEDS) - 1, st["speed"] + 1)
            elif k == ord("-"):
                st["speed"] = max(0, st["speed"] - 1)

    try:
        curses.wrapper(main)
    finally:
        ended = stop_producer(q, stop, th)
        print(sh.status())
        print("model stopped" if ended else "WARNING: the model thread didn't stop; the process exit will end it")


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
