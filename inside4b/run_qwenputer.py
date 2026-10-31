"""Inside-4B: Qwenputer-4B (Qwen3.5-4B with an RV32 computer compiled into its checkpoint) answers stage-5 questions
with its built-in computer, through upstream's own chat path (chipformer.rv.chat.RvChat.generate: greedy, thinking
off, the calc tool in the chat template, every computer token checked against upstream's reference machine).

Which computer actually runs: with --path executor (the default, and what upstream's benchmark uses), upstream's fast
circuit executor (Executor2) computes the circuit's arithmetic from the same circuit description, in fp32, instead of
running the checkpoint's bf16 computer-layer modules; its every token is checked against the reference machine. This
script does not execute or verify the checkpoint's bf16 computer layers on that path (qwenputer-v100 showed the two
give identical text on 200 problems, but that isn't re-checked here).

Budgets: at most --max-new text tokens and --max-chip computer trace tokens per question, cumulative (upstream's
counters). Upstream's reference check (the lock) would otherwise stop at its default 1,000,000 records per call; it
gets the same allowance as the trace budget, so it never binds first (a record is at least one trace token).

Rows are stage 5's replies.jsonl format (graded by stage5/score.py); a lock error is a trace_mismatch (hard failure).
usage (riscv dir on PYTHONPATH, the qwenputer venv, the V100 visible, inside the V100 lock):
    python run_qwenputer.py MODEL_DIR QUESTIONS.jsonl OUT_DIR [--path executor] [--max-new 8192] [--max-chip 2000000]
"""
import argparse, functools, hashlib, json, os, sys, time
from pathlib import Path

import torch

from chipformer.rv import bridge as BR
from chipformer.rv import calc
from chipformer.rv import chat as CH
from chipformer.rv import elf, ref

sys.path.insert(0, str(Path(__file__).parent))
import frames  # noqa: E402

FIRMWARE = Path(CH.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def python_reference(expr):
    """Upstream's exact Python value for an expression the computer called an error (scorer: limit vs bad input)."""
    try:
        return calc.expected(str(expr).replace("\n", "").strip(" ")).strip()
    except Exception:  # noqa: BLE001
        return None


def per_call_trace(tokens, v0, tcc):
    """Trace (chip) tokens after each </tool_call>, read off the generated ids."""
    counts, cur = [], None
    for t in tokens:
        if t == tcc:
            counts.append(0)
            cur = len(counts) - 1
        elif t >= v0 and cur is not None:
            counts[cur] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--path", choices=["executor"], default="executor")
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--max-chip", type=int, default=2_000_000,
                    help="computer trace tokens per question, cumulative (upstream's default is 60,000)")
    ap.add_argument("--arm", default="C")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows_path = os.path.join(a.out, "replies.jsonl")
    if os.path.exists(rows_path):
        sys.exit(f"{rows_path} exists: never overwrite a run")
    Path(a.out, "questions.sha256").write_text(sha(a.questions) + f"  {a.questions}\n")
    t0 = time.time()
    chat = CH.RvChat(a.model, executor=True)
    info = {"arm": a.arm, "path": "upstream fast circuit executor (fp32), checked against the RV32 reference; the "
            "checkpoint's bf16 computer layers are not executed", "load_seconds": round(time.time() - t0, 1),
            "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            "tokenizer_sha256": sha(Path(a.model) / "tokenizer.json"), "firmware_sha256": sha(FIRMWARE),
            "max_new": a.max_new, "max_chip": a.max_chip, "lock_max_records_per_call": a.max_chip,
            "allow_bf16_reduced_precision": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    Path(a.out, "startup.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"loaded in {info['load_seconds']} s", flush=True)
    qs = [json.loads(l) for l in open(a.questions) if l.strip()]
    torch.cuda.reset_peak_memory_stats()
    for n, q in enumerate(qs):
        ids = chat.prompt_ids([{"role": "user", "content": q["prompt"]}])
        m = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(m)
        m2 = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(m2)
        lock = BR.Session(chat.T, m2)
        lock.history(ids)
        lock.qwen = functools.partial(lock.qwen, max_records=a.max_chip)   # the checker never caps first
        t = time.time()
        err = None
        try:
            r = chat.generate(ids, m, a.max_new, max_chip=a.max_chip, check_head=True, lock=lock)
        except Exception as e:  # noqa: BLE001  (kept as a harness error for this question)
            r, err = {"tokens": []}, f"{type(e).__name__}: {e}"
        secs = round(time.time() - t, 2)
        toks = r["tokens"]
        text = chat.text_of(toks) if toks else ""
        ended = bool(toks) and toks[-1] in chat.eos
        stop = ("eos" if ended else "lock error" if r.get("lock_error") else
                "computer budget" if (r.get("chip_tokens") or 0) >= a.max_chip else
                "text budget" if (r.get("text_tokens") or 0) >= a.max_new else None if err else "other")
        fr = frames.parse(text)
        traces = per_call_trace(toks, chat.v0, chat.T.TCC)
        calls = []
        for i, c in enumerate(fr["calls"]):
            if c["response"] is None:
                continue
            rec = {"expression": c["expression"], "call_text": c["call_text"], "result": c["response"], "error": None,
                   "trace_tokens": traces[i] if i < len(traces) else None}
            if str(c["response"]).startswith("error"):
                rec["python_reference"] = python_reference(c["expression"])
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
    info["gpu_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    Path(a.out, "startup.json").write_text(json.dumps(info, indent=1) + "\n")
    print("done")


if __name__ == "__main__":
    main()
