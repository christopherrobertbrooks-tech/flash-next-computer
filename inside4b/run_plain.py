"""Inside-4B: the plain Qwen3.5-4B (the base Qwenputer was built from) on stage-5 questions, in the same Hugging Face
runtime and decoding as run_qwenputer.py: greedy, thinking off, the same chat template and prompt ids, one token per
step, exact bf16 matmul reductions (chipformer.q35.fast.exact_matmuls, as upstream's RvChat sets them).

  --arm A   alone: no tools in the prompt.
  --arm R   beside, RISC-V: the calc tool in the prompt (as Qwenputer has it). At </tool_call>, upstream's Python RV32
            reference machine runs the same calc firmware (calc-bridge.elf) on the call's tokens, and **every text
            token it writes** goes into the stream (the HALT-echo newline included, as Qwenputer's circuit writes it).
  --arm B   beside, ordinary Python: firmware-style framing (the firmware's capture of the expression, ported exactly,
            and the same response frame including the final newline), but the value comes from upstream's Python
            calculator (calc.expected). It doesn't model the firmware's register/stack limits ("too complex").

Budgets, the same for R and Qwenputer: at most --max-new text tokens per question (tool-response tokens included, as
upstream counts), and at most --max-trace computer trace tokens per question, cumulative over its calls (Qwenputer's
max_chip counts the same thing). R's reference machine also gets a per-call record allowance equal to what's left of
that budget (each record is one or more trace tokens, so it never binds first).

Rows are stage 5's replies.jsonl format (graded by stage5/score.py).
usage (riscv dir on PYTHONPATH, the qwenputer venv, V100 visible, inside the V100 lock):
    python run_plain.py BASE_MODEL_DIR QUESTIONS.jsonl OUT_DIR --arm A|B|R [--max-new 8192] [--max-trace 2000000]
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from chipformer.q35.fast import exact_matmuls
from chipformer.rv import bridge as BR
from chipformer.rv import calc
from chipformer.rv import chat as CH
from chipformer.rv import elf, ref
from chipformer.rv import spec as S

sys.path.insert(0, str(Path(__file__).parent))
import frames  # noqa: E402

EOS = {248046, 248044}
DEVICE = "cuda"                                            # test_streams.py drives generate() on the CPU
FIRMWARE = Path(calc.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---- arm R: the reference machine runs the firmware ---------------------------------------------------------
def emulator_response(T, call_ids, max_records):
    """(every text token the machine writes after </tool_call>, trace tokens, records). call_ids: <tool_call> ...
    </tool_call>. Raises RuntimeError if the run doesn't halt within max_records."""
    m = ref.Machine.from_program(elf.program(FIRMWARE))
    BR.park(m)
    steps0 = m.steps                                        # parking ran some instructions: count only this call's
    sess = BR.Session(T, m)
    k = len(sess.toks)
    sess.qwen(call_ids, max_records=max_records)
    after = sess.toks[k + len(call_ids):]                   # everything the run pushed after </tool_call>
    text = [t - S.TEXT_TOKEN for t in after if S.is_text(t)]
    trace = sum(1 for t in after if not S.is_text(t))
    return text, trace, m.steps - steps0                    # instructions executed (refresh records aren't counted)


# ---- arm B: the firmware's capture and frame, Python's value -------------------------------------------------
def capture_rows(T, call_tokens):
    """The text buffer as Session.qwen fills it: row k (1..CAP_MAX) holds the k-th token after <tool_call> (16 bytes,
    NUL-padded); later tokens all overwrite row CAP_MAX."""
    rows = [bytes(S.TEXT_ROW)] * S.TEXT_ROWS
    for cap, q in enumerate(call_tokens, start=1):
        rows[min(cap, S.CAP_MAX)] = T.row_bytes(q)
    return rows


def firmware_expression(T, call_tokens):
    """calc.c's gather(), ported exactly: (expression, error). Rows 1..80; a row stops at its first NUL; an empty row
    ends the call; the value is what follows the second '>' up to the next '<', newlines dropped, at most 255 chars;
    no '<' after all 80 rows -> "too long"."""
    rows = capture_rows(T, call_tokens)
    n, gt, out = 0, 0, []
    i = 1
    while i <= S.CAP_MAX:
        r = rows[i]
        c = r[0]
        if not c:
            break
        k = 0
        while k < S.TEXT_ROW and c:
            if gt < 2:
                gt += c == ord(">")
            elif c == ord("<"):
                return bytes(out).decode(errors="replace"), None
            elif c != ord("\n") and n < 255:
                out.append(c)
                n += 1
            k += 1
            c = r[k] if k < S.TEXT_ROW else 0
        i += 1
    if gt < 2:
        return None, "error"
    if i > S.CAP_MAX:
        return None, "error (too long: split it)"
    return bytes(out).decode(errors="replace"), None


def response_tokens(T, value_line):
    """The firmware's response, token by token, as Session.run pushes it: printable[b] per byte, specials 256-259, and
    the HALT-echo newline at the end."""
    P = T.printable
    TR, TRE, TH, THE = P[256], P[257], P[258], P[259]
    return ([P[10], TR, P[10]] + [P[b] for b in value_line.encode()]
            + [P[10], TRE, P[10], TH, P[10], P[10], THE, P[10]] + [T.NL])


def python_value(expr):
    try:
        return calc.expected(expr.strip(" ")).strip()
    except Exception:  # noqa: BLE001
        return "error"


# ---- generation ----------------------------------------------------------------------------------------------
@torch.no_grad()
def generate(model, ids, max_new, max_trace, arm, T):
    """Greedy, one token per forward step, KV cache. Returns (generated ids, text tokens, calls, stop reason,
    trace tokens used)."""
    out = model(input_ids=torch.tensor([ids], device=DEVICE), use_cache=True)
    cache = out.past_key_values
    gen, n_text, calls, forced, trace_used = [], 0, [], [], 0
    nxt = int(torch.argmax(out.logits[0, -1]))
    stop = None
    while True:
        if n_text >= max_new:
            stop = "text budget"
            break
        gen.append(nxt)
        if not forced and nxt in EOS:                        # like upstream's generate: EOS isn't counted
            stop = "eos"
            break
        n_text += 1
        nxt_after = None
        if forced:
            nxt_after = forced.pop(0)
        elif arm in ("B", "R") and nxt == T.TCC:
            ts = time.time()
            start = max(i for i, t in enumerate(gen) if t == T.TCO) if T.TCO in gen else len(gen) - 1
            call_ids = gen[start:]                          # <tool_call> ... </tool_call>
            cap_expr, _ = firmware_expression(T, call_ids[1:-1])     # what the firmware captures, for every arm
            rec = {"call_text": T.decode_bytes(call_ids[1:-1]).decode(errors="replace"), "expression": cap_expr,
                   "error": None}
            if arm == "R":
                try:
                    forced, trace, records = emulator_response(T, call_ids, max(1, max_trace - trace_used))
                except RuntimeError as e:                    # did not halt within what's left of the budget
                    forced, trace, records = [], max_trace - trace_used + 1, None
                    rec.update(error=f"emulator: {e}", error_kind="budget")
                trace_used += trace
                rec.update(trace_tokens=trace, instructions=records)
                rtext = T.decode_bytes(forced).decode(errors="replace")
                rec["result"] = (rtext.split("<tool_response>")[-1].split("</tool_response>")[0].strip()
                                 if "<tool_response>" in rtext else None)
            else:
                expr, err = firmware_expression(T, call_ids[1:-1])
                value = err if err else python_value(expr)
                forced = response_tokens(T, value)
                rec.update(expression=expr, result=value)
            if rec.get("result") and rec["result"].startswith("error"):   # what Python says, so the scorer can tell
                fx, ferr = firmware_expression(T, call_ids[1:-1])          # a CPU limit from a bad expression
                rec["python_reference"] = ferr or python_value(fx)
            rec["ms"] = round((time.time() - ts) * 1000, 3)
            calls.append(rec)
            if trace_used >= max_trace or not forced:       # like upstream's loop: reaching the cap stops
                stop = "computer budget"
                break
            forced = list(forced)
            nxt_after = forced.pop(0)
        o = model(input_ids=torch.tensor([[nxt]], device=DEVICE), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        nxt = nxt_after if nxt_after is not None else int(torch.argmax(o.logits[0, -1]))
    return gen, n_text, calls, stop, trace_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--arm", choices=["A", "B", "R"], required=True)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--max-trace", type=int, default=2_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows_path = os.path.join(a.out, "replies.jsonl")
    if os.path.exists(rows_path):
        sys.exit(f"{rows_path} exists: never overwrite a run")
    Path(a.out, "questions.sha256").write_text(sha(a.questions) + f"  {a.questions}\n")
    exact_matmuls()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    idx = json.load(open(Path(a.model, "model.safetensors.index.json")))["weight_map"]
    name = "model.language_model.layers.0.linear_attn.A_log"
    with safe_open(str(Path(a.model, idx[name])), "pt") as f:
        assert torch.equal(f.get_tensor(name).cuda(), model.model.layers[0].linear_attn.A_log), "weights not loaded"
    T = BR.Text(Path(a.model) / "tokenizer.json")
    info = {"arm": a.arm, "load_seconds": round(time.time() - t0, 1),
            "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
            "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            "tokenizer_sha256": sha(Path(a.model) / "tokenizer.json"), "firmware_sha256": sha(FIRMWARE),
            "max_new": a.max_new, "max_trace": a.max_trace,
            "allow_bf16_reduced_precision": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    Path(a.out, "startup.json").write_text(json.dumps(info, indent=1) + "\n")
    tools = [CH.CALC_TOOL] if a.arm in ("B", "R") else None
    qs = [json.loads(l) for l in open(a.questions) if l.strip()]
    torch.cuda.reset_peak_memory_stats()
    for n, q in enumerate(qs):
        ids = tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], tools=tools,
                                      add_generation_prompt=True, enable_thinking=False, tokenize=True)
        ids = [int(x) for x in (ids if isinstance(ids, list) else ids["input_ids"])]
        t = time.time()
        err = None
        try:
            gen, n_text, calls, stop, trace_used = generate(model, ids, a.max_new, a.max_trace, a.arm, T)
        except Exception as e:  # noqa: BLE001
            gen, n_text, calls, stop, trace_used, err = [], 0, [], None, 0, f"{type(e).__name__}: {e}"
        secs = round(time.time() - t, 2)
        text = tok.decode(gen, skip_special_tokens=False)
        fr = frames.parse(text)
        row = {"id": q["id"], "group": q["group"], "arm": a.arm, "http": 200 if err is None else None,
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
    info["gpu_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    Path(a.out, "startup.json").write_text(json.dumps(info, indent=1) + "\n")
    print("done")


if __name__ == "__main__":
    main()
