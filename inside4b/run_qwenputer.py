"""Inside-4B pilot: Qwenputer-4B (Qwen3.5-4B with the RV32 computer compiled into its weights) answers stage-5
questions with its built-in computer, through upstream's own chat path (chipformer.rv.chat.RvChat.generate: greedy,
thinking off, calc tool in the chat template, every circuit token checked against a reference machine). Rows are
written in stage 5's replies.jsonl format, so the frozen stage5/score.py grades them unchanged.

usage (riscv dir on PYTHONPATH, the qwenputer venv, the V100 visible, inside the V100 lock):
    python run_qwenputer.py MODEL_DIR QUESTIONS.jsonl OUT_DIR [--path executor|hf] [--max-new 8192] [--max-chip 2000000]
--path executor: upstream's fast executor computes the circuit layers (the default in upstream's benchmark);
--path hf: every layer, the computer's included, runs as the model's own bf16 modules (slower; "fully inside").
"""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path

from chipformer.rv import bridge as BR
from chipformer.rv import calc
from chipformer.rv import chat as CH
from chipformer.rv import elf, ref


def python_reference(expr):
    """Upstream's exact Python value for an expression the computer called an error (scorer: limit vs bad input)."""
    try:
        return calc.expected(str(expr).replace("\n", "").strip(" ")).strip()
    except Exception:  # noqa: BLE001
        return None

FIRMWARE = Path(CH.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--path", choices=["executor", "hf"], default="executor")
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--arm", default="C")
    ap.add_argument("--max-chip", type=int, default=2_000_000,
                    help="computer steps allowed per answer (upstream's default is 60,000; the emulator arm gets 2M)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows_path = os.path.join(a.out, "replies.jsonl")
    if os.path.exists(rows_path):
        sys.exit(f"{rows_path} exists: never overwrite a run")
    Path(a.out, "questions.sha256").write_text(
        hashlib.sha256(open(a.questions, "rb").read()).hexdigest() + f"  {a.questions}\n")
    t0 = time.time()
    chat = CH.RvChat(a.model, executor=(a.path == "executor"))
    Path(a.out, "startup.txt").write_text(f"model loaded in {time.time() - t0:.1f} s (path {a.path})\n")
    print(f"loaded in {time.time() - t0:.1f} s, path {a.path}", flush=True)
    qs = [json.loads(l) for l in open(a.questions) if l.strip()]
    for n, q in enumerate(qs):
        ids = chat.prompt_ids([{"role": "user", "content": q["prompt"]}])
        m = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(m)
        m2 = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(m2)
        lock = BR.Session(chat.T, m2)
        lock.history(ids)
        t = time.time()
        err = None
        try:
            r = chat.generate(ids, m, a.max_new, max_chip=a.max_chip, check_head=True, lock=lock)
        except Exception as e:  # noqa: BLE001  (kept as a harness error for this question)
            r, err = {"tokens": []}, f"{type(e).__name__}: {e}"
        secs = round(time.time() - t, 2)
        text = chat.text_of(r["tokens"]) if r["tokens"] else ""
        ended = bool(r["tokens"]) and r["tokens"][-1] in chat.eos
        final = re.split(r"</tool_response>", text)[-1]                       # the reply after the last tool frame
        final = final.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        calls = re.findall(r"<parameter=expression>\s*(.*?)\s*</parameter>", text, re.S)
        results = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", text, re.S)
        row = {"id": q["id"], "group": q["group"], "arm": a.arm, "http": 200 if err is None else None,
               "seconds": secs, "harness_error": err,
               "raw": {"choices": [{"finish_reason": "stop" if ended else "length",
                                    "message": {"content": final, "full_text": text}}]},
               "tool_calls": [{"expression": e, "result": res, "error": None,
                               **({"python_reference": python_reference(e)} if str(res).startswith("error") else {})}
                              for e, res in zip(calls, results)]
                             + ([{"expression": None, "result": None, "error": str(r.get("lock_error")),
                                  "error_kind": "trace_mismatch"}] if r.get("lock_error") else []),
               "tool_calls_open": [],
               "inside": {k: r.get(k) for k in ("text_tokens", "chip_tokens", "head_mismatch", "lock_error",
                                                "verify_mismatch", "chip_seconds", "text_seconds")},
               "time": time.strftime("%X")}
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        ins = row["inside"]
        print(f"{n + 1:3}/{len(qs)} {q['id']:16} {secs:7.1f}s {'stop' if ended else 'LENGTH'} calls {len(calls)} "
              f"chip {ins['chip_tokens']} head_mismatch {ins['head_mismatch']} lock_error {ins['lock_error']}"
              f"{' ERROR ' + err if err else ''}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
