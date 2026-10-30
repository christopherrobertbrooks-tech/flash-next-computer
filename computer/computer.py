"""The exact computer as a callable service: Qwenputer's RV32 circuit on upstream's Triton executor (fp32 circuit
arithmetic), driven without any language model. A call feeds the tool-call text exactly as the model would write it;
the circuit runs the calc firmware and writes the response; every circuit token is checked against upstream's
reference machine (bridge.Session), and the call stops at the first token that differs.

Fail-closed rules (after Codex's stage 2-3 review and its re-check of a7ac5b3):
- the expression may not contain protocol markers or special tokens, and the whole call must fit the circuit's
  capture buffer (spec.CAP_MAX tokens between <tool_call> and </tool_call>);
- while the call is being fed, neither the reference machine nor the circuit may produce anything; the run starts
  only at the final </tool_call>, and the whole reference stream from there is checked token by token;
- success needs exactly one complete <tool_response>...</tool_response> and a handback exactly where the reference
  hands back; a result is returned only on success (never partial text);
- the budget caps the reference machine's records and the circuit's tokens; max_seconds is one deadline from the
  moment call() is entered, checked while waiting for the computer, before each input token, before each reference
  record and before each circuit step (cooperative: one record or one step can run past it, not more);
- input refusals change nothing; past them the session is marked invalid before any state changes and is valid
  again only after a fully successful call or reset, so a returned error, a raised exception or a failed reset all
  leave call() refusing until new_session() succeeds;
- one instance is one machine: calls are serialized with a lock (never share an instance between conversations).

Built only from the Qwenputer checkpoint's config.json, chipformer.json and tokenizer files (no weights loaded).
Needs upstream's experimental/riscv on PYTHONPATH.
"""
import copy, json, re, threading, time
from pathlib import Path

import torch

from chipformer.q35.ir import Target
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S
from chipformer.rv.tritonexec import Executor2

CALL = "<tool_call>\n<function=calc>\n<parameter=expression>\n{expr}\n</parameter>\n</function>\n</tool_call>"
MARKERS = ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<function", "</function",
           "<parameter", "</parameter", "<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")


class _Late(Exception):
    """The deadline passed inside the reference run."""


class CallRefused(Exception):
    """The call was not run (invalid session or input)."""


class Computer:
    def __init__(self, model_dir, firmware, device="cuda"):
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "chipformer.json").read_text())
        cfg = json.loads((model_dir / "config.json").read_text())
        tc = cfg.get("text_config", cfg)
        self.T = BR.Text(model_dir / "tokenizer.json")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.c = core.build(Target(tc["linear_num_key_heads"], tc["linear_num_value_heads"],
                                   tc["intermediate_size"]), self.T)
        self.v0 = self.meta["chip_token_offset"]
        self.ncode = len(self.meta["code_dims"])
        self.dev = device
        self.firmware = elf.program(firmware)
        self.ex = Executor2(self.c, 1, device)
        self.heads = [st for st in self.ex.steps if st["kind"] == "head"]
        self.keep_ir = torch.tensor(self.c.keep, device=device)
        self.xbuf = torch.zeros(1, self.ex.D, device=device)
        self.kbuf = torch.zeros(len(self.c.keep), device=device)
        self.emb_cache, self.graph, self.lock = {}, None, None
        self.valid = False
        self.mutex = threading.Lock()
        self.first_special = min(i for i in self.tok.all_special_ids if i > 1000)   # Qwen's specials start here

    # --- the executor, as upstream's chat.ExecRow drives it ------------------------------------------------
    def _emb(self, ids):
        new = sorted({i for i in ids if i not in self.emb_cache})
        if new:
            rows = self.ex.embed([i - self.v0 if i >= self.v0 else BR.text_token(i) for i in new])
            for i, r in zip(new, rows):
                self.emb_cache[i] = r
        return torch.stack([self.emb_cache[i] for i in ids])

    def _body(self):
        self.ex.step(self.xbuf)
        self.kbuf.copy_(self.xbuf[0, self.keep_ir])

    @torch.no_grad()
    def _capture(self):
        saved = [(st["S"].clone(), st["hist"].clone()) for st in self.heads]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._body()
        for st, (S_, h) in zip(self.heads, saved):
            st["S"].copy_(S_)
            st["hist"].copy_(h)

    @torch.no_grad()
    def _step(self, mid):
        self.xbuf.copy_(self._emb([mid]))
        self.graph.replay()
        return self.kbuf

    def _token(self, code):
        """The model id the circuit predicts next, or None where the language model would."""
        code = code.float()
        if not bool((code != 0).any()):
            return None
        bits = (code > 0).long().tolist()
        nf = core.NFAM
        t = core.decode_code(sum(b << i for i, b in enumerate(bits[:nf])),
                             sum(b << i for i, b in enumerate(bits[nf:])), self.T)
        if t < 0:
            raise RuntimeError("the circuit predicts no token")
        return t - S.TEXT_TOKEN if S.is_text(t) else self.v0 + t

    def _stream(self, mid):
        return mid - self.v0 if mid >= self.v0 else BR.text_token(mid)

    def _name(self, mid):
        st = self._stream(mid)
        return S.name(st) if st < S.VOCAB else repr(self.tok.decode([mid]))

    # --- the service -----------------------------------------------------------------------------------
    @torch.no_grad()
    def new_session(self, prefix="<|im_start|>assistant\n"):
        """A fresh machine (the firmware parked) and a reference machine beside it."""
        with self.mutex:
            self.valid = False                    # a reset that fails part-way leaves no usable session
            self._new_session(prefix)
            self.valid = True

    def _new_session(self, prefix):
        m = ref.Machine.from_program(self.firmware)
        BR.park(m)
        self.lock = BR.Session(self.T, copy.deepcopy(m))
        ids = self.tok.encode(prefix, add_special_tokens=False)
        core.snapshot(self.c, m)
        if self.graph is None:
            self._capture()
        for st in self.heads:
            st["S"].zero_()
            st["hist"].zero_()
        emb = self._emb(ids)
        for t in range(len(ids)):
            self.xbuf.copy_(emb[t:t + 1])
            self.graph.replay()
        self.ex.load_init()
        self.lock.history(ids)

    def _check_input(self, expr, ids):
        if not isinstance(expr, str):
            raise CallRefused("the expression must be a string")
        for mk in MARKERS:
            if mk in expr:
                raise CallRefused(f"the expression contains the protocol marker {mk!r}")
        inner = self.tok.encode(expr, add_special_tokens=False)
        if any(i >= self.first_special for i in inner):
            raise CallRefused("the expression encodes to a special token")
        if ids.count(self.T.TCO) != 1 or ids.count(self.T.TCC) != 1 or ids[-1] != self.T.TCC:
            raise CallRefused("the call must contain exactly one <tool_call> ... </tool_call>")
        captured = len(ids) - ids.index(self.T.TCO) - 2
        if captured > S.CAP_MAX:
            raise CallRefused(f"the call is {captured} tokens; the circuit captures at most {S.CAP_MAX}")

    MAX_SESSION_TOKENS = 2_000_000           # the reference's history grows with every call; start a new session

    @torch.no_grad()
    def call(self, expr, max_chip=200_000, max_seconds=120.0):
        """Run one calc call. Returns {result, response, chip_tokens, seconds, error}: result is set only when error
        is None, i.e. every circuit token matched the reference, there was exactly one complete response, and the
        circuit handed control back exactly where the reference does. Raises CallRefused (nothing run, session
        unchanged) on a busy computer, an invalid session or invalid input. Anything else, a returned error or a raised
        exception, invalidates the session. max_seconds is one deadline from the moment call() is entered: it covers
        the wait for the computer, the input, the reference run and the circuit's output (checked between steps)."""
        deadline = time.monotonic() + max_seconds
        if not self.mutex.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise CallRefused(f"the computer stayed busy for more than {max_seconds} s")
        try:
            if not self.valid:
                raise CallRefused("no valid session: call new_session() first (after any failed call too)")
            if len(self.lock.toks) > self.MAX_SESSION_TOKENS:
                self.valid = False
                raise CallRefused("the session's history is full: call new_session()")
            ids = self.tok.encode(CALL.format(expr=expr) if isinstance(expr, str) else "", add_special_tokens=False)
            self._check_input(expr, ids)          # refusals change nothing
            self.valid = False                    # from here on the state changes: valid again only on success
            out = self._call(expr, ids, max_chip, max_seconds, deadline)
            if out["error"] is None:
                self.valid = True
            return out
        finally:
            self.mutex.release()

    def _call(self, expr, ids, max_chip, max_seconds, deadline):
        t0 = time.monotonic()
        out = {"expr": expr, "result": None, "response": None, "chip_tokens": 0, "error": None}
        late = f"budget: more than {max_seconds} s"

        def fail(why):
            out["error"] = why
            out["seconds"] = round(time.monotonic() - t0, 3)
            return out

        m = self.lock.m                                   # the reference checks the deadline before every record
        record = m.record

        def timed_record():
            if time.monotonic() > deadline:
                raise _Late()
            return record()

        m.record = timed_record
        try:
            out = self._run(ids, max_chip, deadline, out, fail, late)
            out["seconds"] = round(time.monotonic() - t0, 3)
            return out
        except _Late:
            return fail(late + " (in the reference run)")
        finally:
            del m.record

    def _run(self, ids, max_chip, deadline, out, fail, late):

        expect = []
        for j, q in enumerate(ids):                       # the model's tokens: the circuit captures them
            if time.monotonic() > deadline:
                return fail(late + f" (input token {j})")
            last = j == len(ids) - 1
            k = len(self.lock.toks)
            try:
                self.lock.qwen([q], max_records=max_chip)
            except RuntimeError as e:                     # the reference machine did not halt within the budget
                return fail(f"reference: {e} (budget {max_chip} records)")
            added = list(self.lock.toks[k + 1:])
            keep = self._step(q)
            if not last:
                if added:
                    return fail(f"the reference ran before the call closed (input token {j})")
                if self._token(keep[:self.ncode]) is not None:
                    return fail(f"the circuit claimed a position inside the call (input token {j})")
            else:
                expect = added
        if not expect:
            return fail("the reference produced no run at </tool_call>")
        produced = []
        nxt = self._token(keep[:self.ncode])
        while nxt is not None:
            want = expect.pop(0) if expect else None
            if self._stream(nxt) != want:
                return fail({"at": len(produced), "got": self._name(nxt),
                             "want": None if want is None else (S.name(want) if want < S.VOCAB else
                                                                repr(self.tok.decode([want - S.TEXT_TOKEN])))})
            produced.append(nxt)
            if nxt >= self.v0:
                out["chip_tokens"] += 1
                if out["chip_tokens"] > max_chip:
                    return fail(f"budget: more than {max_chip} computer tokens")
            if time.monotonic() > deadline:
                return fail(late)
            keep = self._step(nxt)
            nxt = self._token(keep[:self.ncode])
        if expect:
            return fail(f"handed back early: {len(expect)} reference tokens left")
        text = self.tok.decode([t for t in produced if t < self.v0], skip_special_tokens=False)
        frames = re.findall(r"<tool_response>\n(.*?)\n</tool_response>", text, re.S)
        if len(frames) != 1 or text.count("<tool_response>") != 1:
            return fail(f"expected exactly one complete response, found {len(frames)}")
        out["response"] = text
        out["result"] = frames[0]
        return out
