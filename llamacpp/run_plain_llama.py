"""LLAMACPP-PLAN v2 stage 1: inside-4b's arm R (plain Qwen3.5-4B with the calc firmware on the RV32 reference machine
beside it) with the model in llama.cpp (the f16 GGUF, through qwen_half_helper's token mode) instead of Hugging Face.
Everything else is inside-4b's run_plain.py, imported unchanged: the prompt ids (the same chat template and tools),
the emulator response (every text token the reference machine writes, the HALT newline included), the firmware
capture, the budgets and their stop rules, the row format (arm "L", graded by stage5/score.py --compare L,R).
generate() below is run_plain.generate with the two model calls replaced (prefill, then one step per token, every
token fed including forced ones, at consecutive text positions); test_run_plain_llama.py checks it gives identical
results to run_plain.generate on scripted models.

Fails closed: a helper error or death records that question's row with harness_error and stops the run (exit 1), so
an incomplete run can't be scored as complete.

usage (riscv dir on PYTHONPATH, the qwenputer venv, V100 visible, inside the V100 lock):
    python run_plain_llama.py BASE_MODEL_DIR GGUF HELPER QUESTIONS.jsonl OUT_DIR [--max-new 8192] [--max-trace 2000000]
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "inside4b" if (HERE / "inside4b").is_dir() else HERE.parent / "inside4b"))  # deployed copy, or the repo's
import frames  # noqa: E402
import run_plain as RP  # noqa: E402
from helper_client import Helper, HelperError  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import chat as CH  # noqa: E402


class LlamaBackend:
    """prefill(ids) / step(id) -> the argmax id, at consecutive text positions from 0."""

    def __init__(self, helper):
        self.h, self.pos = helper, 0

    def prefill(self, ids):
        self.h.reset()
        ids_, _ = self.h.tokens(list(range(len(ids))), ids)
        self.pos = len(ids)
        return ids_[0]

    def step(self, t):
        ids_, _ = self.h.tokens([self.pos], [t])
        self.pos += 1
        return ids_[0]


def generate(be, ids, max_new, max_trace, arm, T):
    """run_plain.generate with the model calls on a backend (kept line for line otherwise)."""
    nxt = be.prefill(ids)
    gen, n_text, calls, forced, trace_used = [], 0, [], [], 0
    stop = None
    while True:
        if n_text >= max_new:
            stop = "text budget"
            break
        gen.append(nxt)
        if not forced and nxt in RP.EOS:
            stop = "eos"
            break
        n_text += 1
        nxt_after = None
        if forced:
            nxt_after = forced.pop(0)
        elif arm in ("B", "R") and nxt == T.TCC:
            ts = time.time()
            start = max(i for i, t in enumerate(gen) if t == T.TCO) if T.TCO in gen else len(gen) - 1
            call_ids = gen[start:]
            cap_expr, _ = RP.firmware_expression(T, call_ids[1:-1])
            rec = {"call_text": T.decode_bytes(call_ids[1:-1]).decode(errors="replace"), "expression": cap_expr,
                   "error": None}
            if arm == "R":
                try:
                    forced, trace, records = RP.emulator_response(T, call_ids, max(1, max_trace - trace_used))
                except RuntimeError as e:
                    forced, trace, records = [], max_trace - trace_used + 1, None
                    rec.update(error=f"emulator: {e}", error_kind="budget")
                trace_used += trace
                rec.update(trace_tokens=trace, instructions=records)
                rtext = T.decode_bytes(forced).decode(errors="replace")
                rec["result"] = (rtext.split("<tool_response>")[-1].split("</tool_response>")[0].strip()
                                 if "<tool_response>" in rtext else None)
            else:
                expr, err = RP.firmware_expression(T, call_ids[1:-1])
                value = err if err else RP.python_value(expr)
                forced = RP.response_tokens(T, value)
                rec.update(expression=expr, result=value)
            if rec.get("result") and rec["result"].startswith("error"):
                fx, ferr = RP.firmware_expression(T, call_ids[1:-1])
                rec["python_reference"] = ferr or RP.python_value(fx)
            rec["ms"] = round((time.time() - ts) * 1000, 3)
            calls.append(rec)
            if trace_used >= max_trace or not forced:
                stop = "computer budget"
                break
            forced = list(forced)
            nxt_after = forced.pop(0)
        m = be.step(nxt)
        nxt = nxt_after if nxt_after is not None else m
    return gen, n_text, calls, stop, trace_used


def gpu_mib(pid):
    """GPU memory the helper process holds (nvidia-smi), MiB; None if not listed."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        p, m = [x.strip() for x in line.split(",")]
        if p == str(pid):
            return int(m)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("gguf"); ap.add_argument("helper")
    ap.add_argument("questions"); ap.add_argument("out")
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--max-trace", type=int, default=2_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows_path = os.path.join(a.out, "replies.jsonl")
    if os.path.exists(rows_path):
        sys.exit(f"{rows_path} exists: never overwrite a run")
    Path(a.out, "questions.sha256").write_text(RP.sha(a.questions) + f"  {a.questions}\n")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    T = BR.Text(Path(a.model) / "tokenizer.json")
    qs = [json.loads(l) for l in open(a.questions) if l.strip()]
    cfg = json.loads(Path(a.model, "config.json").read_text())
    n_vocab = cfg.get("text_config", cfg)["vocab_size"]   # the GGUF's vocabulary (the embedding rows)
    t0 = time.time()
    with open(Path(a.out, "helper-stderr.log"), "w") as herr, \
            Helper(a.helper, a.gguf, 16384, stderr=herr, n_vocab=n_vocab) as h:
        be = LlamaBackend(h)
        be.prefill([1])                                  # wait until the model is loaded
        info = {"arm": "L", "engine": "llama.cpp via qwen_half_helper (token mode)", "load_seconds":
                round(time.time() - t0, 1), "gguf": a.gguf, "gguf_sha256": RP.sha(a.gguf),
                "helper_sha256": RP.sha(a.helper), "tokenizer_sha256": RP.sha(Path(a.model) / "tokenizer.json"),
                "runner_sha256": RP.sha(__file__), "run_plain_sha256": RP.sha(RP.__file__),
                "frames_sha256": RP.sha(frames.__file__), "client_sha256": RP.sha(sys.modules["helper_client"].__file__),
                "firmware_sha256": RP.sha(RP.FIRMWARE), "max_new": a.max_new, "max_trace": a.max_trace,
                "helper_gpu_mib_after_load": gpu_mib(h.p.pid)}
        Path(a.out, "startup.json").write_text(json.dumps(info, indent=1) + "\n")
        samples = [info["helper_gpu_mib_after_load"]]      # sampled after each question; None = query failed
        failed = None
        for n, q in enumerate(qs):
            ids = tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], tools=[CH.CALC_TOOL],
                                          add_generation_prompt=True, enable_thinking=False, tokenize=True)
            ids = [int(x) for x in (ids if isinstance(ids, list) else ids["input_ids"])]
            t = time.time()
            err = None
            try:
                gen, n_text, calls, stop, trace_used = generate(be, ids, a.max_new, a.max_trace, "R", T)
            except Exception as e:  # noqa: BLE001
                gen, n_text, calls, stop, trace_used, err = [], 0, [], None, 0, f"{type(e).__name__}: {e}"
            secs = round(time.time() - t, 2)
            samples.append(gpu_mib(h.p.pid))
            text = tok.decode(gen, skip_special_tokens=False)
            fr = frames.parse(text)
            row = {"id": q["id"], "group": q["group"], "arm": "L", "http": 200 if err is None else None,
                   "seconds": secs, "harness_error": err,
                   "raw": {"choices": [{"finish_reason": "stop" if stop == "eos" else "length",
                                        "message": {"content": fr["final"], "full_text": text}}]},
                   "tool_calls": calls,
                   "tool_calls_open": [c for c in fr["calls"] if c["response"] is None],
                   "run": {"stop": stop, "text_tokens": n_text, "trace_tokens": trace_used,
                           "prompt_tokens": len(ids),
                           "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                           "frames_seen": len(fr["calls"])},
                   "time": time.strftime("%X")}
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{n + 1:3}/{len(qs)} {q['id']:16} {secs:7.1f}s {stop} calls {len(calls)} text {n_text} "
                  f"trace {trace_used}{' ERROR ' + err if err else ''}", flush=True)
            if err:
                failed = err
                break
        ok = [m for m in samples if m is not None]
        info["helper_gpu_mib_sampled_max"] = max(ok) if ok else None
        info["helper_gpu_mib_samples_unavailable"] = len(samples) - len(ok)
        Path(a.out, "startup.json").write_text(json.dumps(info, indent=1) + "\n")
    if failed:
        sys.exit(f"stopped: {failed}")
    print("done")


if __name__ == "__main__":
    main()
