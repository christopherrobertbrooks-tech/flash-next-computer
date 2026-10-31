"""LLAMACPP-PLAN v2 stages 2-3: Qwenputer answers stage-5 questions with its Qwen half in llama.cpp (the mixed
F16/F32 GGUF, through qwen_half_helper's row mode) and its computer on upstream's fp32 executor, every chip token
checked by a lock Session exactly as inside-4b's run_qwenputer.py does. The rows are run_qwenputer's format (arm "S",
graded by stage5/score.py), built by its own helpers (imported, unchanged).

Differences from run_qwenputer.py: split_chat.split_generate instead of RvChat.generate (see split_chat.py);
check_head isn't available (head_mismatch None). Memory (Codex, stage-1 review): a sampler polls nvidia-smi every
0.5 s for this process and the helper on the V100, sums them per sample and keeps the maximum of the sums, plus the
per-process maxima, the baseline, after-load values and how many samples failed (never counted as 0).

Fails closed: a helper error records that question's row as a harness error and stops the run (exit 1).
usage (riscv dir on PYTHONPATH, the qwenputer venv, the V100 visible, inside the V100 lock):
    python run_split.py MODEL_DIR GGUF HELPER QUESTIONS.jsonl OUT_DIR [--max-new 8192] [--max-chip 2000000]
"""
import argparse, functools, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "inside4b" if (HERE / "inside4b").is_dir() else HERE.parent / "inside4b"))
import frames  # noqa: E402
import run_qwenputer as RQ  # noqa: E402
import split_chat as SC  # noqa: E402
from helper_client import Helper  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import elf, ref  # noqa: E402


class MemSampler(threading.Thread):
    """nvidia-smi samples of the given pids' GPU memory (MiB): max of the per-sample sums, per-pid maxima, failures."""

    def __init__(self, pids, every=0.5):
        super().__init__(daemon=True)
        self.pids, self.every, self.stop = [str(p) for p in pids], every, threading.Event()
        self.max_sum, self.max_each, self.n, self.failed = None, {}, 0, 0

    def sample(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            got = {p.strip(): int(m) for p, m in (l.split(",") for l in out.splitlines() if l.strip())}
        except Exception:  # noqa: BLE001
            self.failed += 1
            return None
        vals = [got.get(p) for p in self.pids]
        if any(v is None for v in vals):
            self.failed += 1
            return None
        self.n += 1
        s = sum(vals)
        self.max_sum = s if self.max_sum is None else max(self.max_sum, s)
        for p, v in zip(self.pids, vals):
            self.max_each[p] = max(self.max_each.get(p, 0), v)
        return s

    def run(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.every)

    def report(self):
        return {"sampled_max_sum_mib": self.max_sum, "sampled_max_each_mib": self.max_each, "samples": self.n,
                "samples_unavailable": self.failed, "every_s": self.every,
                "method": "nvidia-smi compute-apps used_memory, both processes, summed per sample; a sampled "
                          "footprint, not an allocation peak"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("gguf"); ap.add_argument("helper")
    ap.add_argument("questions"); ap.add_argument("out")
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--max-chip", type=int, default=2_000_000)
    ap.add_argument("--arm", default="S")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows_path = os.path.join(a.out, "replies.jsonl")
    if os.path.exists(rows_path):
        sys.exit(f"{rows_path} exists: never overwrite a run")
    Path(a.out, "questions.sha256").write_text(RQ.sha(a.questions) + f"  {a.questions}\n")
    import torch
    t0 = time.time()
    chat = SC.SplitChat(a.model)
    me = os.getpid()
    base = MemSampler([me]); base.sample()
    with open(Path(a.out, "helper-stderr.log"), "w") as herr, \
            Helper(a.helper, a.gguf, 16384, stderr=herr, n_vocab=chat.v0) as h:
        h.reset()
        h.rows([0], SC.to_f32(torch.zeros(1, chat.hidden)))             # wait until the model is loaded
        mem = MemSampler([me, h.p.pid]); mem.sample()
        after_load = mem.max_sum
        info = {"arm": a.arm, "path": "Qwen half in llama.cpp (qwen_half_helper row mode, mixed F16/F32 GGUF); the "
                "computer on upstream's fp32 executor (Executor2), checked against the RV32 reference; the "
                "checkpoint's bf16 computer layers are not executed; no HF checkpoint loaded",
                "load_seconds": round(time.time() - t0, 1), "gguf": a.gguf, "gguf_sha256": RQ.sha(a.gguf),
                "helper_sha256": RQ.sha(a.helper), "tokenizer_sha256": RQ.sha(Path(a.model) / "tokenizer.json"),
                "firmware_sha256": RQ.sha(RQ.FIRMWARE), "runner_sha256": RQ.sha(__file__),
                "split_chat_sha256": RQ.sha(SC.__file__), "run_qwenputer_sha256": RQ.sha(RQ.__file__),
                "client_sha256": RQ.sha(sys.modules["helper_client"].__file__),
                "max_new": a.max_new, "max_chip": a.max_chip, "lock_max_records_per_call": a.max_chip,
                "gpu_mib_baseline_parent": base.max_sum, "gpu_mib_after_load_sum": after_load}
        Path(a.out, "startup.json").write_text(json.dumps(info, indent=1) + "\n")
        mem.start()
        torch.cuda.reset_peak_memory_stats()
        qs = [json.loads(l) for l in open(a.questions) if l.strip()]
        failed = None
        for n, q in enumerate(qs):
            ids = chat.prompt_ids([{"role": "user", "content": q["prompt"]}])
            m = ref.Machine.from_program(elf.program(RQ.FIRMWARE))
            BR.park(m)
            m2 = ref.Machine.from_program(elf.program(RQ.FIRMWARE))
            BR.park(m2)
            lock = BR.Session(chat.T, m2)
            lock.history(ids)
            lock.qwen = functools.partial(lock.qwen, max_records=a.max_chip)
            t = time.time()
            err = None
            try:
                r = SC.split_generate(chat, h, ids, m, a.max_new, max_chip=a.max_chip, lock=lock)
            except Exception as e:  # noqa: BLE001
                r, err = {"tokens": []}, f"{type(e).__name__}: {e}"
            secs = round(time.time() - t, 2)
            toks = r["tokens"]
            text = chat.text_of(toks) if toks else ""
            ended = bool(toks) and toks[-1] in chat.eos
            stop = ("eos" if ended else "lock error" if r.get("lock_error") else
                    "computer budget" if (r.get("chip_tokens") or 0) >= a.max_chip else
                    "text budget" if (r.get("text_tokens") or 0) >= a.max_new else None if err else "other")
            fr = frames.parse(text)
            traces = RQ.per_call_trace(toks, chat.v0, chat.T.TCC)
            calls = []
            for i, c in enumerate(fr["calls"]):
                if c["response"] is None:
                    continue
                rec = {"expression": c["expression"], "call_text": c["call_text"], "result": c["response"],
                       "error": None, "trace_tokens": traces[i] if i < len(traces) else None}
                if str(c["response"]).startswith("error"):
                    rec["python_reference"] = RQ.python_reference(c["expression"])
                calls.append(rec)
            if r.get("lock_error"):
                calls.append({"expression": None, "result": None, "error": str(r["lock_error"]),
                              "error_kind": "trace_mismatch"})
            row = {"id": q["id"], "group": q["group"], "arm": a.arm, "http": 200 if err is None else None,
                   "seconds": secs, "harness_error": err,
                   "raw": {"choices": [{"finish_reason": "stop" if ended else "length",
                                        "message": {"content": fr["final"], "full_text": text}}]},
                   "tool_calls": calls,
                   "tool_calls_open": [c for c in fr["calls"] if c["response"] is None],
                   "run": {"stop": stop, "prompt_tokens": len(ids),
                           "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                           "frames_seen": len(fr["calls"])},
                   "inside": {k: r.get(k) for k in ("text_tokens", "chip_tokens", "head_mismatch", "lock_error",
                                                    "verify_mismatch", "chip_seconds", "text_seconds")},
                   "time": time.strftime("%X")}
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            ins = row["inside"]
            print(f"{n + 1:3}/{len(qs)} {q['id']:16} {secs:7.1f}s {stop} calls {len(calls)} chip {ins['chip_tokens']} "
                  f"lock_error {ins['lock_error']}{' ERROR ' + err if err else ''}", flush=True)
            if err:
                failed = err
                break
        mem.stop.set()
        mem.join(5)
        info["memory"] = mem.report()
        info["parent_torch_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        Path(a.out, "startup.json").write_text(json.dumps(info, indent=1) + "\n")
    if failed:
        sys.exit(f"stopped: {failed}")
    print("done")


if __name__ == "__main__":
    main()
