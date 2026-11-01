"""LLAMACPP-PLAN v2 stage 2: Qwenputer with its Qwen half in llama.cpp and its computer on upstream's fp32 executor.

SplitChat is upstream's RvChat on the executor path without the 18.8 GB HF checkpoint: the same tokenizer, circuit
(core.build), executor (ExecRow / Executor2) and decoding helpers (code_token, circuit_token, prompt_ids, text_of),
and `_exec_hc` built from the checkpoint's own embedding rows (the text ids, the first 2,560 dims) exactly as
upstream builds it. split_generate() is upstream's RvChat.generate (executor path, fast off) copied line for line,
with its two Qwen calls moved to the llama.cpp helper:
  - the prompt: instead of the full-checkpoint prefill, the executor-built prompt rows (stage 0b: bit-identical to
    the HF prefill rows on all 12 fixtures) are sent as one prefill, and the first token is the helper's argmax;
  - each text step: instead of `_qwen_step(hc, cache, pos)`, the same hc row goes to the helper at text position pos.
Everything else (the firmware snapshot, the executor steps, circuit ownership, the lock, the budgets, EOS) is
upstream's. Two upstream diagnostics need full-vocabulary logits the helper doesn't return, so they're not
available: check_head (a circuit-owned token vs Qwen's own unmasked argmax) is reported as None.
test_split_chat.py checks split_generate against upstream's generate on scripted fakes (CPU).
"""
import json, time
from pathlib import Path

import torch

from chipformer.rv import bridge as BR
from chipformer.rv import chat as CH
from chipformer.rv import core
from chipformer.rv import spec as S


class SplitChat(CH.RvChat):
    """RvChat(path, executor=True) without loading the HF model (see the module docstring)."""

    def __init__(self, path, device="cuda"):                    # noqa: super().__init__ deliberately not called
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer

        from chipformer.q35.fast import exact_matmuls
        from chipformer.q35.ir import Target

        exact_matmuls()
        self.path = Path(path)
        self.meta = json.loads((self.path / "chipformer.json").read_text())
        self.dev = device
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.T = BR.Text(self.path / "tokenizer.json")
        cfg = AutoConfig.from_pretrained(self.path)
        cfg = getattr(cfg, "text_config", cfg)
        self.c = core.build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.intermediate_size), self.T)
        self.n_chip, self.v0 = self.meta["chip_layers"], self.meta["chip_token_offset"]
        self.code_dims = torch.tensor(self.meta["code_dims"], device=self.dev)
        self.eos = {248046, 248044}
        self.hidden = cfg.hidden_size
        self.fast, self.executor, self.verify = False, True, False
        d0 = self.meta["chip_dims_offset"]
        idx = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        name = "model.embed_tokens.weight"
        with safe_open(str(self.path / idx[name]), "pt") as f:
            self.emb_text = f.get_slice(name)[: self.v0, :d0].to(self.dev)   # the only embedding values _exec_hc reads
        self._build_exec()

    def _exec_hc(self, ids, keep):
        """Upstream's _exec_hc, reading the same embedding values (the text rows' first d0 dims)."""
        hc = torch.zeros(self.hidden, dtype=torch.bfloat16, device=self.dev)
        hc[: self.d0] = self.emb_text[ids]
        hc[self.keep_cols] = keep.to(torch.bfloat16)
        return hc


@torch.no_grad()
def exec_prompt_rows(chat, ids):
    """ExecRow.start (zero states and history, the prompt through the graph, then load_init keeping the lag
    history), also returning each prompt position's Qwen input row built by _exec_hc."""
    er = chat.erow
    if er.graph is None:
        er._capture()
    for st in er.heads:
        st["S"].zero_()
        st["hist"].zero_()
    emb = chat._exec_emb(ids)
    rows = []
    for t in range(len(ids)):
        er.xbuf.copy_(emb[t:t + 1])
        er.graph.replay()
        rows.append(chat._exec_hc(ids[t], er.kbuf))
    er.ex.load_init()
    return torch.stack(rows)


def to_f32(hc):
    return hc.float().cpu().numpy()


@torch.no_grad()
def split_generate(self, helper, ids, machine, max_new=4000, max_chip=60000, lock=None, on_token=None,
                   prompt_rows=exec_prompt_rows, verbose=False, lm_input="rows"):
    """Upstream's RvChat.generate (executor path) with the prompt prefill and the Qwen steps on `helper`.
    Lines differing from upstream are marked SPLIT.

    lm_input "rows" (stages 2-3): the helper runs Qwenputer's widened Qwen half fed the rows Qwen's layers would get.
    lm_input "tokens" (FAST-PLAN): the helper runs a stock Qwen fed token ids: the prompt at 0..P-1, then every
    processed text input (forced echoes and the HALT newline included) at P, P+1, ...; chip tokens and EOS are never
    sent. The executor's prompt initialization is upstream's own ExecRow.start, and everything on the computer side
    (snapshot, executor steps, _exec_hc for circuit_token, ownership, the lock, budgets) is unchanged."""
    assert lm_input in ("rows", "tokens"), lm_input
    core.snapshot(self.c, machine)
    helper.reset()                                                      # SPLIT: fresh caches and position
    if lm_input == "rows":
        rows = prompt_rows(self, ids)                                   # SPLIT: the prompt's rows (and ExecRow.start)
        pids, _ = helper.rows(list(range(len(ids))), to_f32(rows))      # SPLIT: the prefill on the helper
    else:
        self._exec_start(ids)                                           # SPLIT: upstream's ExecRow.start, as is
        pids, _ = helper.tokens(list(range(len(ids))), ids)             # SPLIT: the prompt's ids on the helper
    pos = len(ids)  # the next text token's RoPE position
    gen, n_text, n_chip_tok = [], 0, 0
    # the prompt's last position: Qwen predicts (nothing captured yet)
    nxt = int(pids[0])                                                  # SPLIT: the helper's argmax (text vocabulary)
    t0 = time.time()
    t_chip = t_text = 0.0
    expect, lock_error = [], None
    while n_text < max_new and n_chip_tok < max_chip:
        gen.append(nxt)
        if on_token is not None:
            on_token(nxt)
        if lock is not None:  # check against the reference as we go
            st = nxt - self.v0 if nxt >= self.v0 else BR.text_token(nxt)
            if expect:
                e = expect.pop(0)
                if st != e:
                    lock_error = {"at": len(gen) - 1, "got": S.name(st) if st < S.VOCAB
                                  else self.tok.decode([nxt]),
                                  "want": S.name(e) if e < S.VOCAB
                                  else self.tok.decode([e - S.TEXT_TOKEN]),
                                  "recent": [S.name(t - self.v0) if t >= self.v0 else
                                             self.tok.decode([t]) for t in gen[-12:]]}
                    break
            elif st < S.VOCAB:
                lock_error = {"at": len(gen) - 1, "got": S.name(st), "want": "text"}
                break
            else:
                k = len(lock.toks)
                try:
                    lock.qwen([nxt])
                except RuntimeError as e:  # the reference run did not halt
                    lock_error = {"at": len(gen) - 1, "got": str(e), "want": "a halt"}
                    break
                expect = list(lock.toks[k + 1:])
        if nxt in self.eos:
            break
        ts = time.time()
        chip_in = nxt >= self.v0
        keep = self._exec_step(nxt)
        hc = None if chip_in else self._exec_hc(nxt, keep)
        if chip_in:  # a chip token
            n_chip_tok += 1
            nxt = self.code_token(keep[: len(self.code_dims)])
            t_chip += time.time() - ts
            if nxt is None:
                raise RuntimeError("the circuit gave up a chip position")
        else:
            n_text += 1
            qids, _ = (helper.rows([pos], to_f32(hc[None])) if lm_input == "rows"   # SPLIT: Qwen's layers on the
                       else helper.tokens([pos], [nxt]))                        # helper (row or the text id)
            pos += 1
            ct = self.circuit_token(hc)
            nxt = ct if ct is not None else int(qids[0])               # SPLIT: argmax over the text vocabulary
            t_text += time.time() - ts
    dt = time.time() - t0
    return {"tokens": gen, "text_tokens": n_text, "chip_tokens": n_chip_tok,
            "seconds": dt, "head_mismatch": None, "lock_error": lock_error,              # SPLIT: check_head n/a
            "verify_mismatch": None, "chip_seconds": round(t_chip, 2), "text_seconds": round(t_text, 2)}
