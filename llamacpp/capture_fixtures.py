"""LLAMACPP-PLAN v2 stage 0a: capture immutable teacher-forced fixtures on the trusted HF path.

Upstream's own RvChat.generate (executor path, greedy, the calc firmware, every chip token checked by a lock Session as
in inside-4b) answers the 12 fixed questions. Hooks record, without changing what runs:
- every row entering Qwen's first layer (layer n_chip): the prompt's rows from the full-checkpoint prefill, then one
  `_exec_hc` row per text step;
- per Qwen position (the prompt's last, then each text step): the input id, its text (RoPE) position, the masked
  top-5 over the text vocabulary with values, and whether the circuit owns the next token (circuit_token);
- the generated ids, the lock result and the stop reason;
- candidate prompt rows for stage 0b: the prompt run through the executor from zero state (as ExecRow.start does),
  the kept values read after every position and assembled with `_exec_hc`.
One .pt per question plus manifest.json (hashes of firmware, tokenizer, model config/index, code). Refuses to
overwrite an existing output folder.

usage (riscv dir on PYTHONPATH, the qwenputer venv, the V100 visible, inside the V100 lock):
    python capture_fixtures.py MODEL_DIR QUESTIONS.jsonl OUT_DIR
"""
import functools, hashlib, json, os, sys, time
from pathlib import Path

import torch

from chipformer.rv import bridge as BR
from chipformer.rv import chat as CH
from chipformer.rv import elf, ref

FIRMWARE = Path(CH.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"
FIXED = [f"final-{g}-00{k}" for g in (1, 2, 3, 4) for k in (1, 2, 3)]
EXTRA = {"id": "extra-two-calls", "group": "extra",
         "prompt": "First use the calc tool to compute 2**64 // 7. Then make a second calc call to compute 7 - 9. "
                   "Give both exact results."}
MAX_NEW, MAX_CHIP = 8192, 2_000_000


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def top5(logits, v0):
    v, i = torch.topk(logits[:v0].float(), 5)
    return i.tolist(), v.tolist()


@torch.no_grad()
def executor_prompt_rows(chat, ids):
    """Stage 0b's candidate: the prompt through the executor from zero state (ExecRow.start's loop), the kept values
    after each position assembled into Qwen's input row by upstream's own _exec_hc. Leaves the executor in the same
    state ExecRow.start would (the next question's start() zeroes it again)."""
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
        rows.append(chat._exec_hc(ids[t], er.kbuf.clone()).cpu())
    return torch.stack(rows)


def main():
    model_dir, qpath, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=False)
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(qpath) if l.strip()}
    chat = CH.RvChat(model_dir, executor=True)
    rec = {}

    def pre_hook(mod, args, kwargs):                     # the row(s) entering Qwen's first layer
        h = args[0] if args else kwargs["hidden_states"]
        rec["rows"].append(h[0].detach().to("cpu", copy=True))

    chat.lm.layers[chat.n_chip].register_forward_pre_hook(pre_hook, with_kwargs=True)

    def model_hook(mod, args, kwargs, output):           # fires only for the prompt prefill (decode uses lm_head)
        rec["prefill_top5"] = top5(output.logits[0, -1], chat.v0)

    chat.model.register_forward_hook(model_hook, with_kwargs=True)
    real_qwen, real_ct = chat._qwen_step, chat.circuit_token

    def qwen_step(hc, cache, pos):
        logits = real_qwen(hc, cache, pos)
        rec["steps"].append({"gen_index": rec["n_gen"] - 1, "input_id": rec["last_id"], "pos": pos,
                             "top5": top5(logits, chat.v0)})
        return logits

    def circuit_token(hc):
        ct = real_ct(hc)
        if rec["steps"] and "circuit" not in rec["steps"][-1]:
            rec["steps"][-1]["circuit"] = ct              # None: Qwen chooses the next token here
        return ct

    chat._qwen_step, chat.circuit_token = qwen_step, circuit_token
    manifest = {"model_dir": model_dir, "questions": qpath, "questions_sha256": sha(qpath),
                "firmware_sha256": sha(FIRMWARE), "tokenizer_sha256": sha(Path(model_dir) / "tokenizer.json"),
                "config_sha256": sha(Path(model_dir) / "config.json"),
                "chipformer_sha256": sha(Path(model_dir) / "chipformer.json"),
                "index_sha256": sha(Path(model_dir) / "model.safetensors.index.json"),
                "chat_py_sha256": sha(CH.__file__), "script_sha256": sha(__file__),
                "n_chip": chat.n_chip, "v0": chat.v0, "max_new": MAX_NEW, "max_chip": MAX_CHIP, "fixtures": []}

    def run(q):
        ids = chat.prompt_ids([{"role": "user", "content": q["prompt"]}])
        m = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(m)
        m2 = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(m2)
        lock = BR.Session(chat.T, m2)
        lock.history(ids)
        lock.qwen = functools.partial(lock.qwen, max_records=MAX_CHIP)
        rec.clear()
        rec.update(rows=[], steps=[], n_gen=0, last_id=None)

        def on_token(t):
            rec["n_gen"] += 1
            rec["last_id"] = t

        t0 = time.time()
        r = chat.generate(ids, m, MAX_NEW, max_chip=MAX_CHIP, lock=lock, on_token=on_token)
        secs = time.time() - t0
        toks = r["tokens"]
        stop = ("eos" if toks and toks[-1] in chat.eos else "lock error" if r.get("lock_error") else
                "computer budget" if r["chip_tokens"] >= MAX_CHIP else
                "text budget" if r["text_tokens"] >= MAX_NEW else "other")
        prefill = rec["rows"][0]
        assert prefill.shape[0] == len(ids), (prefill.shape, len(ids))
        step_rows = torch.stack([x[-1] for x in rec["rows"][1:]]) if len(rec["rows"]) > 1 else None
        assert step_rows is None or step_rows.shape[0] == len(rec["steps"])
        assert all("circuit" in s for s in rec["steps"])
        cand = executor_prompt_rows(chat, ids)
        calls = sum(1 for t in toks if t == chat.T.TCC)
        fx = {"id": q["id"], "group": q["group"], "prompt": q["prompt"], "prompt_ids": ids, "generated": toks,
              "stop": stop, "lock_error": r.get("lock_error"), "text_tokens": r["text_tokens"],
              "chip_tokens": r["chip_tokens"], "calls": calls, "seconds": round(secs, 2),
              "prefill_rows": prefill, "prefill_top5": rec["prefill_top5"], "step_rows": step_rows,
              "steps": rec["steps"], "executor_prompt_rows": cand}
        torch.save(fx, Path(out) / f"{q['id']}.pt")
        manifest["fixtures"].append({k: fx[k] for k in ("id", "group", "stop", "lock_error", "text_tokens",
                                                        "chip_tokens", "calls", "seconds")}
                                    | {"prompt_len": len(ids), "qwen_positions": 1 + len(rec["steps"]),
                                       "sha256": sha(Path(out) / f"{q['id']}.pt")})
        print(f"{q['id']:16} {stop:10} {secs:6.1f}s text {r['text_tokens']} chip {r['chip_tokens']} calls {calls} "
              f"lock_error {r.get('lock_error')}", flush=True)
        return calls

    calls = [run(qs[i]) for i in FIXED]
    if max(calls) < 2:
        print("none of the 12 has two calls: adding the fixed two-call question (hard checks only)", flush=True)
        run(EXTRA)
    Path(out, "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("done")


if __name__ == "__main__":
    main()
