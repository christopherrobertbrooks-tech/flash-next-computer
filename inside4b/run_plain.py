"""Inside-4B: the plain Qwen3.5-4B (the base Qwenputer was built from: its Qwen weights are unchanged apart from the
norm rescale for the wider residual) on stage-5 questions, in the same Hugging Face runtime and decoding as
run_qwenputer.py: greedy, thinking off, the same chat template and prompt ids, one token per step.

  --arm A   alone: no tools in the prompt.
  --arm R   beside, RISC-V: the same as B, but the value comes from upstream's Python RV32 reference machine running
            the same calc firmware as Qwenputer's circuit (same limits, same response tokens); only where the CPU lives
            differs (a Python program beside the model, not the model's weights).
  --arm B   beside: the calc tool in the prompt (as Qwenputer has it). When the model writes </tool_call>, host Python
            computes the value and the runner writes the response into the stream **exactly as Qwenputer's firmware
            does** (calc.c BRIDGE: one token per byte, "\\n<tool_response>\\nVALUE\\n</tool_response>\\n<think>\\n\\n
            </think>\\n"; the model writes the newline after it). The expression is read as the firmware reads it (what
            follows the second '>' up to the next '<', newlines dropped, at most 80 captured tokens of 16 bytes each),
            then upstream's exact Python calculator (calc.expected, surrounding spaces skipped) gives VALUE. So B
            differs from Qwenputer only in who computes: host Python here, the circuit in the weights there.

Budget: at most --max-new text tokens per question, counting the tool-response tokens, as upstream's generate counts.
Rows are stage 5's replies.jsonl format (graded by the frozen stage5/score.py).

usage (riscv dir on PYTHONPATH, the qwenputer venv, V100 visible, inside the V100 lock):
    python run_plain.py BASE_MODEL_DIR QUESTIONS.jsonl OUT_DIR --arm A|B [--max-new 8192]
"""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from safetensors import safe_open

from chipformer.rv import bridge as BR
from chipformer.rv import calc
from chipformer.rv import chat as CH
from chipformer.rv import elf, ref
from chipformer.rv import spec as S

FIRMWARE = Path(calc.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"
MAX_RECORDS = 2_000_000


def emulator_response(T, call_ids):
    """Arm R: upstream's Python RV32 reference machine runs the same calc firmware on the call, exactly as the circuit
    in Qwenputer's weights would; returns (the response's text tokens it writes, its CPU records)."""
    m = ref.Machine.from_program(elf.program(FIRMWARE))
    BR.park(m)
    sess = BR.Session(T, m)
    k = len(sess.toks)
    sess.qwen(call_ids, max_records=MAX_RECORDS)           # <tool_call> ... </tool_call>: the run starts at the end
    tcc = k + len(call_ids) - 1                            # where </tool_call> sits in the stream
    toks, own = sess.toks[tcc + 1:], sess.own[tcc + 1:]
    text = [t - S.TEXT_TOKEN for t, o in zip(toks, own) if S.is_text(t) and o]   # the circuit-owned text tokens
    records = sum(1 for t in toks if not S.is_text(t))
    return text, records

EOS = {248046, 248044}


def firmware_expression(T, call_tokens):
    """The firmware's gather(): (expression, error) from the tokens between <tool_call> and </tool_call>."""
    n, gt, out = 0, 0, []
    # the capture: row k holds the k-th token after <tool_call>; tokens past 80 all land in row 80 (the last wins)
    rows = call_tokens if len(call_tokens) <= 80 else call_tokens[:79] + call_tokens[-1:]
    for q in rows:                                        # text rows 1..80, 16 bytes each
        for c in T.bytes[q][:16]:
            if gt < 2:
                gt += c == ord(">")
            elif c == ord("<"):
                return bytes(out).decode(errors="replace"), None
            elif c != ord("\n") and n < 255:
                out.append(c)
                n += 1
    if gt < 2:
        return None, "error"
    return (None, "error (too long: split it)") if len(call_tokens) > 80 else (bytes(out).decode(errors="replace"), None)


def response_tokens(T, value_line):
    """The firmware's response, token by token (printable[b] per byte; specials 256-259)."""
    P = T.printable
    TR, TRE, TH, THE = P[256], P[257], P[258], P[259]
    return [P[10], TR, P[10]] + [P[b] for b in value_line.encode()] + [P[10], TRE, P[10], TH, P[10], P[10], THE, P[10]]


@torch.no_grad()
def generate(model, ids, max_new, tool, T):
    """Greedy, one token per forward step, KV cache. tool: run Python at </tool_call> and force the response."""
    out = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=True)
    cache, logits = out.past_key_values, out.logits[0, -1]
    gen, n_text, calls, tool_s = [], 0, [], 0.0
    forced = []
    nxt = int(torch.argmax(logits))
    while n_text < max_new:
        gen.append(nxt)
        n_text += 1
        if not forced and nxt in EOS:
            break
        if forced:
            nxt_after = forced.pop(0)
        elif tool == "R" and nxt == T.TCC:
            ts = time.time()
            start = max(i for i, t in enumerate(gen) if t == T.TCO) if T.TCO in gen else len(gen) - 1
            try:
                forced, records = emulator_response(T, gen[start:])
                rtext = T.decode_bytes(forced).decode(errors="replace")
                value = rtext.split("<tool_response>")[-1].split("</tool_response>")[0].strip() if "<tool_response>" in rtext else rtext.strip()
                err = None
            except RuntimeError as e:                       # the reference run did not halt within MAX_RECORDS
                forced, records, value, err = [], 0, None, f"emulator: {e}"
            tool_s += time.time() - ts
            calls.append({"expression": T.decode_bytes(gen[start + 1:-1]).decode(errors="replace"), "result": value,
                          "error": err, "error_kind": "budget" if err else None, "records": records,
                          "ms": round((time.time() - ts) * 1000, 3)})
            if not forced:
                break
            forced = list(forced)
            nxt_after = forced.pop(0)
        elif tool and nxt == T.TCC:
            ts = time.time()
            start = max(i for i, t in enumerate(gen) if t == T.TCO) if T.TCO in gen else len(gen) - 1
            expr, err = firmware_expression(T, gen[start + 1:-1])
            if err is None:
                try:
                    value = calc.expected(expr.strip(" ")).strip()
                except Exception as e:  # noqa: BLE001
                    value = "error"
            else:
                value = err
            tool_s += time.time() - ts
            calls.append({"expression": expr, "result": value, "error": None, "ms": round((time.time() - ts) * 1000, 3)})
            forced = response_tokens(T, value)
            nxt_after = forced.pop(0)
        else:
            nxt_after = None
        o = model(input_ids=torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        nxt = nxt_after if nxt_after is not None else int(torch.argmax(o.logits[0, -1]))
    return gen, n_text, calls, tool_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--arm", choices=["A", "B", "R"], required=True)
    ap.add_argument("--max-new", type=int, default=8192)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows_path = os.path.join(a.out, "replies.jsonl")
    if os.path.exists(rows_path):
        sys.exit(f"{rows_path} exists: never overwrite a run")
    Path(a.out, "questions.sha256").write_text(
        hashlib.sha256(open(a.questions, "rb").read()).hexdigest() + f"  {a.questions}\n")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    # the weights really loaded (not freshly initialised): compare one tensor with the checkpoint file
    idx = json.load(open(Path(a.model, "model.safetensors.index.json")))["weight_map"]
    name = "model.language_model.layers.0.linear_attn.A_log"
    with safe_open(str(Path(a.model, idx[name])), "pt") as f:
        assert torch.equal(f.get_tensor(name).cuda(), model.model.layers[0].linear_attn.A_log), "weights not loaded"
    T = BR.Text(Path(a.model) / "tokenizer.json")
    Path(a.out, "startup.txt").write_text(f"model loaded in {time.time() - t0:.1f} s, arm {a.arm}, "
                                          f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f} B params, "
                                          f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GiB\n")
    tools = [CH.CALC_TOOL] if a.arm in ("B", "R") else None
    qs = [json.loads(l) for l in open(a.questions) if l.strip()]
    for n, q in enumerate(qs):
        ids = tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], tools=tools,
                                      add_generation_prompt=True, enable_thinking=False, tokenize=True)
        ids = [int(x) for x in (ids if isinstance(ids, list) else ids["input_ids"])]
        t = time.time()
        err = None
        try:
            gen, n_text, calls, tool_s = generate(model, ids, a.max_new, {"A": None, "B": "B", "R": "R"}[a.arm], T)
        except Exception as e:  # noqa: BLE001
            gen, n_text, calls, tool_s, err = [], 0, [], 0.0, f"{type(e).__name__}: {e}"
        secs = round(time.time() - t, 2)
        text = tok.decode(gen, skip_special_tokens=False)
        ended = bool(gen) and gen[-1] in EOS
        final = re.split(r"</tool_response>", text)[-1].replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        row = {"id": q["id"], "group": q["group"], "arm": a.arm, "http": 200 if err is None else None,
               "seconds": secs, "harness_error": err,
               "raw": {"choices": [{"finish_reason": "stop" if ended else "length",
                                    "message": {"content": final, "full_text": text}}]},
               "tool_calls": calls, "tool_calls_open": [],
               "plain": {"text_tokens": n_text, "prompt_tokens": len(ids), "tool_seconds": round(tool_s, 4)},
               "time": time.strftime("%X")}
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{n + 1:3}/{len(qs)} {q['id']:16} {secs:7.1f}s {'stop' if ended else 'LENGTH'} calls {len(calls)} "
              f"text {n_text}{' ERROR ' + err if err else ''}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
