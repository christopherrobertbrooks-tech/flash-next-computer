"""Viewer phase 2 premise check: can plain Qwen3.5-4B write RV32IM assembly that assembles and runs correctly?

Eight small programs with a known result in a0. Each reply's code block is assembled with upstream's assembler and run
on the reference machine (at most 200,000 records); right = it halts via exit with the expected a0. Greedy decoding,
thinking on and off (two passes). Writes one JSON line per (task, thinking) and prints a summary.

usage (riscv dir on PYTHONPATH, the qwenputer venv, V100 visible, inside the V100 lock):
    python asm_premise.py BASE_MODEL_DIR OUT.jsonl
"""
import json, re, sys, time

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

from chipformer.q35.fast import exact_matmuls
from chipformer.rv import asm, ref

RULES = ("Write a complete RV32IM assembly program for this assembler: a `_start:` label, ABI register names, "
         "pseudo-instructions li/mv/j/call/ret/beqz/bnez allowed, `.data`/`.word` allowed, no other directives. "
         "The machine starts with sp set; there is no operating system. End the program with `li a7, 93` then "
         "`ecall`, which exits with the value in a0. Reply with the program in one ```asm code block. Task: ")
TASKS = [
    ("sum", "put the sum 1 + 2 + ... + 10 in a0 using a loop.", 55),
    ("factorial", "put 6! (6 factorial) in a0 using a loop and the mul instruction.", 720),
    ("max3", "put the largest of the values 17, 42 and 8 in a0, using branches.", 42),
    ("array", "store the words 3, 1, 4, 1, 5 in a .data array, then loop over it and put their sum in a0.", 14),
    ("prologue", "write a function `square` that takes a0 and returns a0*a0; it must save ra and s0 on the stack "
                 "in a prologue (addi sp, sp, -16; sw ra, 12(sp); sw s0, 8(sp)) and restore them in an epilogue. "
                 "Call it with 9 from _start and exit with the result.", 81),
    ("fib", "put the 10th Fibonacci number in a0 (fib(1) = fib(2) = 1), using a loop.", 55),
    ("recursive", "write a recursive function `fact` (saving ra on the stack) and exit with fact(5).", 120),
    ("bits", "count the 1 bits in the number 0xF0F0 using shifts and andi, and put the count in a0.", 8),
]


def run_asm(src):
    try:
        prog = asm.assemble(src)
    except Exception as e:  # noqa: BLE001
        return {"assembled": False, "error": f"{type(e).__name__}: {e}"[:200]}
    m = ref.Machine.from_program(prog)
    n = 0
    while not m.halted and n < 200_000:
        m.record()
        n += 1
    return {"assembled": True, "halted": m.halted, "fault": m.fault, "exit": m.exit_code, "a0": m.x[10],
            "records": n}


def main():
    model_dir, out = sys.argv[1:3]
    exact_matmuls()
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3_5ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    rows = []
    for think in (False, True):
        for name, task, want in TASKS:
            ids = tok.apply_chat_template([{"role": "user", "content": RULES + task}], add_generation_prompt=True,
                                          enable_thinking=think, return_tensors="pt", return_dict=True)
            t = time.time()
            with torch.no_grad():
                o = model.generate(**{k: v.cuda() for k, v in ids.items()}, max_new_tokens=6144, do_sample=False)
            reply = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            blocks = re.findall(r"```(?:asm|assembly|s|riscv)?\s*\n(.*?)```", reply.split("</think>")[-1], re.S)
            res = run_asm(blocks[-1]) if blocks else {"assembled": False, "error": "no code block"}
            ok = res.get("halted") and res.get("fault") is None and res.get("exit") == want
            row = {"task": name, "thinking": think, "want": want, "right": bool(ok), "seconds": round(time.time() - t, 1),
                   **res, "code": blocks[-1] if blocks else None, "reply_tail": reply[-400:]}
            rows.append(row)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{'think' if think else 'no-think':8} {name:10} {'RIGHT' if ok else 'wrong'} "
                  f"{ {k: res.get(k) for k in ('assembled', 'exit', 'a0', 'fault', 'error') if res.get(k) is not None} }",
                  flush=True)
    for think in (False, True):
        r = [x for x in rows if x["thinking"] == think]
        print(f"thinking {'on' if think else 'off'}: {sum(x['right'] for x in r)}/{len(r)} right, "
              f"{sum(x['assembled'] for x in r)}/{len(r)} assembled")


if __name__ == "__main__":
    main()
