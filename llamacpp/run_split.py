"""LLAMACPP-PLAN stages 2-3: Qwenputer answers stage-5 questions with its Qwen half in llama.cpp (qwen_half_helper's
row mode) and its computer on upstream's fp32 executor, every chip token checked by a lock Session exactly as
inside-4b's run_qwenputer.py does. The rows are run_qwenputer's format (arm "S", graded by stage5/score.py
--compare S,C), built by its own helpers (imported, unchanged).

Differences from run_qwenputer.py: split_chat.split_generate instead of RvChat.generate (see split_chat.py);
check_head isn't available (head_mismatch None).

Run validity (Codex stage-2 review): the run stops with a nonzero exit, after saving the question's row, on
- a lock error (a chip token differing from the reference): the design is invalid ("INVALID DESIGN");
- any exception (a helper error, death, timeout, non-finite output, OOM): the run is incomplete;
- a prompt that couldn't fit with the text budget in the helper's context (checked before generating).
With --expect HASHES.json, every listed artifact must match its sha256 before anything loads.

Memory (Codex, stage-1/2 reviews): MemSampler polls `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory`
filtered to the V100's UUID, every 0.5 s from after-load through the helper's shutdown; each sample records its
time, phase and per-pid MiB; it sums the two processes per sample. During "run" a missing pid makes that sample
unavailable (never 0); during "shutdown" the helper's absence is expected (recorded as exited). The sampler is
joined before the report is written. It's a sampled footprint, not an allocation peak.

usage (riscv dir on PYTHONPATH, the qwenputer venv, the V100 visible, inside the V100 lock):
    python run_split.py MODEL_DIR GGUF HELPER QUESTIONS.jsonl OUT_DIR [--max-new 8192] [--max-chip 2000000]
        [--n-ctx 16384] [--expect HASHES.json] [--gpu-uuid UUID] [--lm-input rows|tokens] [--arm S|F|G]

--lm-input tokens (FAST-PLAN arms F, G): the GGUF is a stock Qwen fed token ids; the computer side is unchanged.
"""
import argparse, functools, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "inside4b" if (HERE / "inside4b").is_dir() else HERE.parent / "inside4b"))
import frames  # noqa: E402
import run_qwenputer as RQ  # noqa: E402

V100_UUID = "GPU-9c9e042a-edd4-dbf9-b6a4-c0d88b33a796"


def parse_apps(text, uuid):
    """{pid: MiB} on the GPU `uuid` from `gpu_uuid,pid,used_memory` csv lines (other GPUs' entries ignored)."""
    out = {}
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0] == uuid:
            out[parts[1]] = out.get(parts[1], 0) + int(parts[2])
    return out


def query_apps():
    return subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout


class MemSampler(threading.Thread):
    def __init__(self, uuid, pids, every=0.5, query=query_apps, log=None):
        super().__init__(daemon=True)
        self.uuid, self.pids, self.every, self.query, self.log = uuid, {k: str(v) for k, v in pids.items()}, every, \
            query, log
        self.stop, self.phase, self.samples = threading.Event(), "run", []

    def sample(self):
        t = time.time()
        try:
            got = parse_apps(self.query(), self.uuid)
        except Exception as e:  # noqa: BLE001
            self.samples.append({"t": t, "phase": self.phase, "unavailable": f"query failed: {e}"})
            return
        each = {name: got.get(pid) for name, pid in self.pids.items()}
        rec = {"t": round(t, 3), "phase": self.phase, "mib": each}
        missing = [n for n, v in each.items() if v is None]
        if missing and not (self.phase == "shutdown" and missing == ["helper"]):
            rec["unavailable"] = f"not listed: {missing}"
        else:
            rec["sum"] = sum(v for v in each.values() if v is not None)
        self.samples.append(rec)

    def run(self):
        while not self.stop.is_set():
            t0 = time.monotonic()
            self.sample()
            self.stop.wait(max(0.0, self.every - (time.monotonic() - t0)))   # a fixed cadence, query time included

    def finish(self):
        self.stop.set()
        self.join()                                       # no timeout: the last query (bounded at 10 s) completes
        self.sample()                                     # one final sample after the helper has exited

    def report(self):
        ok = [s for s in self.samples if "sum" in s]
        by = {p: [s["sum"] for s in ok if s["phase"] == p] for p in ("run", "shutdown")}
        each = {n: max((s["mib"][n] for s in ok if s["mib"].get(n) is not None), default=None) for n in self.pids}
        gaps = [b["t"] - a["t"] for a, b in zip(self.samples, self.samples[1:])]
        return {"gpu_uuid": self.uuid, "pids": self.pids, "samples": len(self.samples),
                "samples_unavailable": len(self.samples) - len(ok),
                "sampled_max_sum_mib": max((s["sum"] for s in ok), default=None),
                "sampled_max_sum_mib_by_phase": {p: max(v, default=None) for p, v in by.items()},
                "sampled_max_each_mib": each, "every_s": self.every,
                "actual_interval_s_median": sorted(gaps)[len(gaps) // 2] if gaps else None,
                "method": "nvidia-smi compute-apps used_memory on the V100's UUID, both processes summed per "
                          "sample, through the helper's shutdown; a sampled footprint, not an allocation peak"}


def verify_artifacts(expect_path):
    want = json.loads(Path(expect_path).read_text())["files"]
    bad = [p for p, h in want.items() if not Path(p).exists() or RQ.sha(p) != h]
    if bad:
        raise SystemExit(f"artifact identity check failed: {bad}")
    return len(want)


def run_questions(chat, helper, qs, out, max_new, max_chip, n_ctx, generate, arm="S"):
    """The question loop. Returns (status, reason): status "ok", "invalid design" (a lock error) or "incomplete"
    (an exception or a preflight failure). Every attempted question's row is saved before stopping."""
    rows_path = os.path.join(out, "replies.jsonl")
    for n, q in enumerate(qs):
        ids = chat.prompt_ids([{"role": "user", "content": q["prompt"]}])
        if len(ids) + max_new > n_ctx:
            return "incomplete", f"{q['id']}: prompt {len(ids)} + text budget {max_new} > helper context {n_ctx}"
        m = RQ.ref.Machine.from_program(RQ.elf.program(RQ.FIRMWARE))
        RQ.BR.park(m)
        m2 = RQ.ref.Machine.from_program(RQ.elf.program(RQ.FIRMWARE))
        RQ.BR.park(m2)
        lock = RQ.BR.Session(chat.T, m2)
        lock.history(ids)
        lock.qwen = functools.partial(lock.qwen, max_records=max_chip)
        t = time.time()
        err = None
        try:
            r = generate(chat, helper, ids, m, max_new, max_chip=max_chip, lock=lock)
        except Exception as e:  # noqa: BLE001
            r, err = {"tokens": []}, f"{type(e).__name__}: {e}"
        secs = round(time.time() - t, 2)
        toks = r["tokens"]
        text = chat.text_of(toks) if toks else ""
        ended = bool(toks) and toks[-1] in chat.eos
        stop = ("eos" if ended else "lock error" if r.get("lock_error") else
                "computer budget" if (r.get("chip_tokens") or 0) >= max_chip else
                "text budget" if (r.get("text_tokens") or 0) >= max_new else None if err else "other")
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
        row = {"id": q["id"], "group": q["group"], "arm": arm, "http": 200 if err is None else None,
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
        if r.get("lock_error"):
            return "invalid design", f"{q['id']}: lock error {r['lock_error']}"
        if err:
            return "incomplete", f"{q['id']}: {err}"
    return "ok", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("gguf"); ap.add_argument("helper")
    ap.add_argument("questions"); ap.add_argument("out")
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--max-chip", type=int, default=2_000_000)
    ap.add_argument("--n-ctx", type=int, default=16384)
    ap.add_argument("--arm", default="S")
    ap.add_argument("--expect", help="JSON {files: {path: sha256}}; every one must match before anything loads")
    ap.add_argument("--gpu-uuid", default=V100_UUID)
    ap.add_argument("--lm-input", choices=["rows", "tokens"], default="rows")
    a = ap.parse_args()
    n_checked = verify_artifacts(a.expect) if a.expect else 0
    os.makedirs(a.out, exist_ok=True)
    rows_path = os.path.join(a.out, "replies.jsonl")
    if os.path.exists(rows_path):
        sys.exit(f"{rows_path} exists: never overwrite a run")
    Path(a.out, "questions.sha256").write_text(RQ.sha(a.questions) + f"  {a.questions}\n")
    import torch
    import split_chat as SC
    from helper_client import Helper
    t0 = time.time()
    chat = SC.SplitChat(a.model)
    me = os.getpid()
    base = parse_apps(query_apps(), a.gpu_uuid).get(str(me))
    qs = [json.loads(l) for l in open(a.questions) if l.strip()]
    status, reason, mem = "incomplete", "not started", None
    info = {}
    try:
        with open(Path(a.out, "helper-stderr.log"), "w") as herr:
            h = Helper(a.helper, a.gguf, a.n_ctx, stderr=herr, n_vocab=chat.v0)
            try:
                h.reset()
                if a.lm_input == "rows":                                   # wait until the model is loaded
                    h.rows([0], SC.to_f32(torch.zeros(1, chat.hidden)))
                else:
                    h.tokens([0], [chat.T.TCO])
                mem = MemSampler(a.gpu_uuid, {"parent": me, "helper": h.p.pid})
                mem.sample()
                lm = ("Qwen half in llama.cpp (qwen_half_helper row mode)" if a.lm_input == "rows" else
                      "a stock Qwen in llama.cpp (qwen_half_helper token mode)")
                info = {"arm": a.arm, "lm_input": a.lm_input, "path": f"{lm}; the computer on upstream's fp32 "
                        "executor (Executor2), checked against the RV32 reference; the checkpoint's bf16 computer "
                        "layers are not executed; no HF checkpoint loaded",
                        "load_seconds": round(time.time() - t0, 1), "gguf": a.gguf, "gguf_sha256": RQ.sha(a.gguf),
                        "helper_sha256": RQ.sha(a.helper), "tokenizer_sha256": RQ.sha(Path(a.model) / "tokenizer.json"),
                        "firmware_sha256": RQ.sha(RQ.FIRMWARE), "runner_sha256": RQ.sha(__file__),
                        "split_chat_sha256": RQ.sha(SC.__file__), "run_qwenputer_sha256": RQ.sha(RQ.__file__),
                        "client_sha256": RQ.sha(sys.modules["helper_client"].__file__),
                        "artifacts_verified": n_checked, "max_new": a.max_new, "max_chip": a.max_chip,
                        "lock_max_records_per_call": a.max_chip, "n_ctx": a.n_ctx,
                        "gpu_mib_parent_before_helper": base, "gpu_mib_after_load": mem.samples[-1]}
                Path(a.out, "startup.json").write_text(json.dumps(info, indent=1) + "\n")
                mem.start()
                torch.cuda.reset_peak_memory_stats()
                gen = functools.partial(SC.split_generate, lm_input=a.lm_input)
                status, reason = run_questions(chat, h, qs, a.out, a.max_new, a.max_chip, a.n_ctx, gen, a.arm)
            finally:
                if mem is not None:
                    mem.phase = "shutdown"
                rc = h.close()
                if status == "ok" and rc != 0:
                    status, reason = "incomplete", f"helper exited with code {rc}"
    finally:
        if mem is not None:
            mem.finish()
            info["memory"] = mem.report()
            Path(a.out, "memory-samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in mem.samples))
        info["parent_torch_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        info["status"], info["reason"] = status, reason
        Path(a.out, "startup.json").write_text(json.dumps(info, indent=1) + "\n")
    if status != "ok":
        sys.exit(f"{'INVALID DESIGN' if status == 'invalid design' else 'INCOMPLETE'}: {reason}")
    print("done")


if __name__ == "__main__":
    main()
