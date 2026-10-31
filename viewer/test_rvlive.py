"""CPU-only controls for rvlive.Shadow (no model, no GPU): feed it token streams built by upstream's reference Session
(exactly what a correct circuit produces) and broken copies of them. Exits 1 if any control fails.

usage (riscv dir on PYTHONPATH): python test_rvlive.py QWENPUTER_MODEL_DIR
"""
import argparse, contextlib, io, queue, sys, threading, time, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rvlive as L  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import elf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

CALL = "<tool_call>\n<function=calc>\n<parameter=expression>\n{e}\n</parameter>\n</function>\n</tool_call>"


def stream(T, prompt, pieces):
    """The stream a correct circuit gives: model text pieces and calls, run through upstream's reference Session."""
    m = ref.Machine.from_program(elf.program(L.FIRMWARE))
    BR.park(m)
    sess = BR.Session(T, m)
    sess.history(prompt)
    k = len(sess.toks)
    for kind, text in pieces:
        sess.qwen(T.encode(CALL.format(e=text) if kind == "call" else text))
    return sess.toks[k:]


FIELDS = ("pc", "steps", "halted", "fault", "exit_code")


def state(m):
    return tuple(getattr(m, f) for f in FIELDS) + (tuple(m.x), tuple(m.ram), tuple(map(tuple, m.bn)), bytes(m.out))


class Witness:
    """An independent reference machine (its own bridge Session, so it captures the call text the firmware reads),
    stepped one record per commit the shadow reports and parked after HALT as the bridge does. The shadow's
    displayed state must equal it at every point of the stream."""

    def __init__(self, T, prompt):
        self.T = T
        self.m = ref.Machine.from_program(elf.program(L.FIRMWARE))
        BR.park(self.m)
        self.sess = BR.Session(T, self.m)
        self.sess.history(prompt)
        self.in_call = self.go = False

    def feed(self, t, committed):
        if committed:
            if not self.go:
                self.m.halted, self.go = False, True
            if self.m.record()[-1] == S.tok("HALT"):
                self.in_call = self.go = False
                if self.m.fault is None:
                    self.m.halted = False
        elif S.is_text(t) and not self.in_call:
            self.sess.qwen([t - S.TEXT_TOKEN], run=False)
            self.in_call = t - S.TEXT_TOKEN == self.T.TCC


def main():
    T = BR.Text(Path(sys.argv[1]) / "tokenizer.json")
    prompt = T.encode("<|im_start|>user\nWhat is it?<|im_end|>\n<|im_start|>assistant\n")
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    # a correct stream: one call
    toks = stream(T, prompt, [("text", "Let me compute. "), ("call", "123456789*987654321"), ("text", "ANSWER: 1")])
    sh = L.Shadow(T, prompt, log_path=None)
    commits = sum(sh.feed(t) == "committed" for t in toks)
    check(f"one call: verified, {commits} records committed, {sh.stream_checked} stream tokens checked", sh.verified and commits > 1000)
    check(f"one call: the shadow printed {bytes(sh.m.out)!r}", b"121932631112635269" in bytes(sh.m.out))
    check("one call: back to text after the call", not sh.in_call and sh.calls == 1)

    # two calls in one answer (the machine parks between them)
    toks = stream(T, prompt, [("call", "2+2"), ("text", "and "), ("call", "(2**64 - 1) // 7"), ("text", " done")])
    sh = L.Shadow(T, prompt)
    for t in toks:
        sh.feed(t)
    check(f"two calls: verified, output {bytes(sh.m.out)!r}", sh.verified and sh.calls == 2
          and b"4" in bytes(sh.m.out) and b"2635249153387078802" in bytes(sh.m.out))

    # one wrong computer token in the middle: flagged at that record, the shadow stops, the text continues
    toks = stream(T, prompt, [("call", "123456789*987654321"), ("text", "ANSWER: 1")])
    chips = [i for i, t in enumerate(toks) if not S.is_text(t)]
    i = chips[len(chips) // 2]
    broken = list(toks)
    broken[i] = toks[i] + 1 if toks[i] + 1 < S.VOCAB else toks[i] - 1
    log = Path("/tmp/rvlive-test-mismatch.jsonl")
    log.unlink(missing_ok=True)
    sh = L.Shadow(T, prompt, log_path=str(log))
    seen_at = None
    for n, t in enumerate(broken):
        sh.feed(t)
        if not sh.verified and seen_at is None:
            seen_at, rec_at = n, sh.records
    check(f"mismatch: flagged at the changed token (position {seen_at}, changed {i})", seen_at == i)
    check(f"mismatch: the shadow stopped ({sh.records} records, frozen at {rec_at})", sh.records == rec_at)
    check(f"mismatch: status says UNVERIFIED ({sh.status()[:40]}...)", sh.status().startswith("UNVERIFIED since record"))
    check("mismatch: logged", log.exists() and log.read_text().count("\n") == 1)
    check("mismatch: the chat text still streams", T.decode_bytes(sh.text_ids).decode().endswith("ANSWER: 1"))

    # a computer token where the model should be writing text
    sh = L.Shadow(T, prompt)
    sh.feed(BR.text_token(T.encode("Hi")[0]))
    sh.feed(S.tok("GO"))
    check("a computer token outside a call is flagged", not sh.verified and "outside" in sh.mismatch["why"])

    # the screen
    toks = stream(T, prompt, [("call", "84/18")])
    sh = L.Shadow(T, prompt)
    for t in toks[: len(toks) // 2]:
        sh.feed(t)
    lines = L.V.text_of(L.V.render(sh, 30, 120, 2), 120)
    check("screen: shadow label and VERIFIED status visible", any("SHADOW" in x for x in lines)
          and any("VERIFIED" in x for x in lines))

    # the screen shows only committed (fully checked) state: compare it with an independent witness at every token
    toks = stream(T, prompt, [("text", "Sure. "), ("call", "99999*99999"), ("text", "then "), ("call", "7-9"),
                              ("text", "done")])

    def lockstep(sh, toks, stop_at=None):
        """Feeds toks; returns (tokens where the display differed from the witness, tokens fed inside a record)."""
        w, diff, inside = Witness(T, prompt), 0, 0
        for n, t in enumerate(toks):
            if stop_at is not None and n == stop_at:
                break
            w.feed(t, sh.feed(t) == "committed")
            inside += sh.pending is not None
            diff += state(sh.m) != state(w.m)
        return diff, inside, w

    sh = L.Shadow(T, prompt)
    diff, inside, w = lockstep(sh, toks)
    check(f"display: equals the witness at all {len(toks)} tokens ({inside} fed mid-record)", diff == 0 and inside > 0)
    check(f"display: ends at the reference result {bytes(sh.m.out)!r}", b"9999800001" in bytes(sh.m.out)
          and b"-2" in bytes(sh.m.out))
    class OldShadow(L.Shadow):                                  # the instrument can fail: the old design showed
        def _commit(self):                                      # the checker machine itself, ahead of the check
            super()._commit()
            self.m = self.cm

    sh = OldShadow(T, prompt)
    sh.m = sh.cm
    diff, inside, _ = lockstep(sh, toks)
    check(f"display control: showing the checker machine is caught ({diff} differing tokens, {inside} mid-record)",
          diff >= inside // 2)

    # after a wrong token the display stays at the last verified state
    chips = [i for i, t in enumerate(toks) if not S.is_text(t)]
    i = chips[len(chips) // 3]
    broken = list(toks)
    broken[i] = toks[i] + 1 if toks[i] + 1 < S.VOCAB else toks[i] - 1
    sh = L.Shadow(T, prompt)
    diff, _, w = lockstep(sh, broken)
    check(f"mismatch display: still the last verified state ({diff} differing tokens, record {sh.records})",
          diff == 0 and not sh.verified)
    sh.finish("eos")
    check("mismatch display: status stays UNVERIFIED after the end", sh.status().startswith("UNVERIFIED"))

    # how the answer ended: only a complete EOS outside a call is VERIFIED
    def cut_after(pred):
        return next(n for n, t in enumerate(toks) if pred(n, t)) + 1

    cuts = {"after </tool_call>": cut_after(lambda n, t: t == BR.text_token(T.TCC)),
            "after GO": cut_after(lambda n, t: t == S.tok("GO")),
            "a BYTE before its echo": cut_after(lambda n, t: not S.is_text(t) and S.family(t)[0] == "BYTE"),
            "HALT before its newline": cut_after(lambda n, t: t == S.tok("HALT"))}
    probe = L.Shadow(T, prompt)
    for n, t in enumerate(toks):
        probe.feed(t)
        if probe.pending is not None and probe.expect and not S.is_text(probe.expect[0]):
            cuts["inside a multi-token record"] = n + 1
            break
    for label, n in cuts.items():
        sh = L.Shadow(T, prompt)
        diff, _, _ = lockstep(sh, toks, stop_at=n)
        sh.finish("eos")
        check(f"EOF {label} (token {n}): {sh.status()[:60]}", sh.status().startswith("INCOMPLETE") and diff == 0)
    sh = L.Shadow(T, prompt)
    lockstep(sh, toks)
    check(f"not finished yet: {sh.status()[:40]}", sh.status().startswith("VERIFIED so far"))
    for reason, want in (("eos", "VERIFIED, answer complete"), ("text budget", "INCOMPLETE (text budget)"),
                         ("error", "INCOMPLETE (error)"), ("stopped", "INCOMPLETE (stopped)")):
        sh.finish(reason)
        check(f"whole stream, ended by {reason}: {sh.status()[:45]}", sh.status().startswith(want))

    # the producer: a full queue, a stop while paused (nothing consumed), an error, a normal end
    def flood(on_token):
        for k in range(100_000):
            on_token(k)
        return {"reason": "eos"}

    q, stop, th = L.make_producer(flood, q_size=8)
    time.sleep(0.3)
    full = q.full()
    t0 = time.time()
    ended = L.stop_producer(q, stop, th, timeout=5)
    check(f"producer: a full queue doesn't block a stop (stopped in {time.time() - t0:.2f} s)", full and ended)

    def boom(on_token):
        on_token(1)
        raise RuntimeError("cuda went away")

    q, stop, th = L.make_producer(boom)
    th.join(5)
    items = []
    while not q.empty():
        items.append(q.get())
    check(f"producer: an error ends with a done item {items[-1]}", items[-1][0] == "done"
          and items[-1][1]["reason"] == "error" and "cuda went away" in items[-1][1]["error"])

    q, stop, th = L.make_producer(lambda on_token: [on_token(k) for k in range(5)] and {"reason": "eos"})
    th.join(5)
    items = [q.get() for _ in range(6)]
    check("producer: a normal end passes every token, then done/eos", items[:5] == list(range(5))
          and items[5] == ("done", {"reason": "eos"}) and not th.is_alive())

    # EOF while the model is still writing a call (after <tool_call>, before </tool_call>) is not a complete answer.
    # Sweep every prefix: the shadow must count itself unfinished exactly from <tool_call> to the HALT newline.
    TCO, TCC = BR.text_token(T.TCO), BR.text_token(T.TCC)

    def call_regions(toks):
        out = []
        for i, t in enumerate(toks):
            if t == TCO:
                h = next(j for j in range(i, len(toks)) if toks[j] == S.tok("HALT"))
                out.append((i, h + 1))                          # h + 1: the HALT's newline echo
        return out

    def sweep(sh, toks):
        regions, wrong = call_regions(toks), 0
        for n, t in enumerate(toks, 1):                         # n tokens fed
            sh.feed(t)
            inside = any(a + 1 <= n <= b for a, b in regions)
            wrong += sh.unfinished != inside
        return wrong, regions

    sh = L.Shadow(T, prompt)
    wrong, regions = sweep(sh, toks)
    check(f"EOF sweep: unfinished exactly inside the {len(regions)} calls at all {len(toks)} prefixes ({wrong} wrong)",
          wrong == 0 and len(regions) == 2)

    class NoArmed(L.Shadow):                                    # control: the predicate before this fix
        @property
        def unfinished(self):
            return self.in_call or self.pending is not None or bool(self.expect)

    wrong, _ = sweep(NoArmed(T, prompt), toks)
    check(f"EOF sweep control: ignoring the call being written is caught ({wrong} wrong prefixes)", wrong > 0)
    tco = toks.index(TCO)
    tcc = toks.index(TCC)
    for label, n in (("right after <tool_call>", tco + 1), ("mid-parameter", (tco + tcc) // 2),
                     ("just before </tool_call>", tcc)):
        sh = L.Shadow(T, prompt)
        for t in toks[:n]:
            sh.feed(t)
        sh.finish("eos")
        check(f"EOF {label} (token {n}): {sh.status()[:45]}", sh.status().startswith("INCOMPLETE"))

    # the real headless() with a fake model: exit status, and cleanup when the viewer side fails
    def fake_start(raw, fail_feed=False):
        chat = types.SimpleNamespace(T=T, v0=10**9)

        def start(*a):
            def gen(on_token):
                for r in raw:
                    on_token(r)
                return {"reason": "eos"}
            q, stop, th = L.make_producer(gen, q_size=8)
            fake_start.last = (q, stop, th)
            return chat, prompt, q, stop, th
        return start

    def raw_of(stream_toks):
        return [t - S.TEXT_TOKEN if S.is_text(t) else 10**9 + t for t in stream_toks]

    args = argparse.Namespace(model=None, question=None, max_new=0, max_chip=0, log=None, rows=30, cols=120)

    def run_headless(raw):
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                L.headless(args, start=fake_start(raw))
            except SystemExit as e:
                return e.code

    check("headless: a complete two-call answer exits 0", run_headless(raw_of(toks)) == 0)
    check("headless: an answer ending right after <tool_call> exits 1", run_headless(raw_of(toks[:tco + 1])) == 1)
    real_feed = L.Shadow.feed

    def bad_feed(self, st):
        if self.stream_checked > 50:
            raise RuntimeError("viewer bug")
        return real_feed(self, st)

    L.Shadow.feed = bad_feed
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            L.headless(args, start=fake_start(raw_of(toks) * 30))
        raised = False
    except RuntimeError:
        raised = True
    finally:
        L.Shadow.feed = real_feed
    q, stop, th = fake_start.last
    th.join(2)
    check(f"headless: a viewer-side error stops the model (raised {raised}, stop {stop.is_set()}, "
          f"alive {th.is_alive()})", raised and stop.is_set() and not th.is_alive())

    # the TUI with a fake terminal: quit before the end is INCOMPLETE (stopped), and the model thread ends
    class Scr:
        def __init__(self, keys):
            self.keys = list(keys)

        def getmaxyx(self):
            return 30, 120

        def getch(self):
            return self.keys.pop(0) if self.keys else -1

        def erase(self): pass
        def refresh(self): pass
        def timeout(self, ms): pass
        def addnstr(self, *a): pass

    def run_tui(keys, raw):
        fake = types.SimpleNamespace(A_NORMAL=0, A_DIM=0, A_BOLD=0, A_REVERSE=0, error=Exception,
                                     curs_set=lambda n: None, use_default_colors=lambda: None,
                                     wrapper=lambda f: f(Scr(keys)))
        saved = sys.modules.get("curses")
        sys.modules["curses"] = fake
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                L.tui(args, start=fake_start(raw))
        finally:
            if saved is not None:
                sys.modules["curses"] = saved
            else:
                sys.modules.pop("curses")
        return out.getvalue(), fake_start.last[2]

    text, th = run_tui([-1, -1, ord("q")], raw_of(toks) * 30)
    check(f"tui: quit mid-answer prints {text.splitlines()[0][:40]!r}, model thread ended {not th.is_alive()}",
          text.startswith("INCOMPLETE (stopped") and not th.is_alive())
    print("LIVE CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
