"""The exact computer as a callable service: Qwenputer's RV32 circuit on upstream's Triton executor (fp32 circuit
arithmetic), driven without any language model. A call feeds the tool-call text exactly as the model would write it;
the circuit runs the calc firmware and writes the response; every circuit token is checked against upstream's
reference machine (bridge.Session), and the call stops at the first token that differs.

Built only from the Qwenputer checkpoint's config.json, chipformer.json and tokenizer files (no weights loaded).
Needs upstream's experimental/riscv on PYTHONPATH.
"""
import copy, json, re, time
from pathlib import Path

import torch

from chipformer.q35.ir import Target
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref
from chipformer.rv import spec as S
from chipformer.rv.tritonexec import Executor2

CALL = "<tool_call>\n<function=calc>\n<parameter=expression>\n{expr}\n</parameter>\n</function>\n</tool_call>"


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

    @torch.no_grad()
    def call(self, expr, max_chip=200_000):
        """Run one calc call. Returns {result, response, chip_tokens, seconds, error}; error is None only if
        every circuit token matched the reference and the circuit handed control back."""
        t0 = time.time()
        ids = self.tok.encode(CALL.format(expr=expr), add_special_tokens=False)
        out = {"expr": expr, "result": None, "response": None, "chip_tokens": 0, "error": None}
        expect = []
        for q in ids:                                   # the model's tokens: the circuit captures them
            k = len(self.lock.toks)
            try:
                self.lock.qwen([q])
            except RuntimeError as e:                   # the reference machine did not halt
                out["error"] = f"reference: {e}"
                return out
            expect = list(self.lock.toks[k + 1:])
            keep = self._step(q)
        produced = []
        nxt = self._token(keep[:self.ncode])
        while nxt is not None:
            want = expect.pop(0) if expect else None
            if self._stream(nxt) != want:
                out["error"] = {"at": len(produced), "got": self._name(nxt),
                                "want": None if want is None else (S.name(want) if want < S.VOCAB else
                                                                   repr(self.tok.decode([want - S.TEXT_TOKEN])))}
                break
            produced.append(nxt)
            if nxt >= self.v0:
                out["chip_tokens"] += 1
                if out["chip_tokens"] > max_chip:
                    out["error"] = f"budget: more than {max_chip} computer tokens"
                    break
            keep = self._step(nxt)
            nxt = self._token(keep[:self.ncode])
        if out["error"] is None and expect:
            out["error"] = f"handed back early: {len(expect)} reference tokens left"
        text = self.tok.decode([t for t in produced if t < self.v0], skip_special_tokens=False)
        out["response"] = text
        m = re.search(r"<tool_response>\n(.*?)\n</tool_response>", text, re.S)
        out["result"] = m.group(1) if m else None
        out["seconds"] = round(time.time() - t0, 3)
        return out
