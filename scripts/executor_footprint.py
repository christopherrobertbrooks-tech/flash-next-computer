"""Stage 3: the Qwenputer computer's fast executor (Triton Executor2, fp32 circuit arithmetic) on its own on the V100:
GPU memory, exactness against the reference circuit, and speed. No language model is loaded (the Qwenputer variant
is built from the checkpoint's config.json and tokenizer.json only).

usage (in qwenputer/experimental/riscv, V100 visible): python executor_footprint.py MODEL_DIR [PROGRAM]
"""
import json, sys, time
from pathlib import Path

import torch

from chipformer.q35.ir import Target
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref
from chipformer.rv.fastexec import timed2
from chipformer.rv.measure import DATA
from chipformer.rv.tritonexec import Executor2, check

MiB = 2 ** 20


def used():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / MiB


def main():
    model = Path(sys.argv[1])
    prog_name = sys.argv[2] if len(sys.argv) > 2 else "c-crc32"
    print("device", torch.cuda.get_device_name(), "| torch", torch.__version__, flush=True)
    before = used()
    torch.zeros(1, device="cuda")
    ctx = used() - before
    print(f"CUDA context of this process: {ctx:.0f} MiB (a separate process pays this once)", flush=True)
    cfg = json.loads((model / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    builds = [("bare RV32 core", lambda: core.build()),
              ("Qwenputer circuit (core + text bridge)",
               lambda: core.build(Target(tc["linear_num_key_heads"], tc["linear_num_value_heads"],
                                         tc["intermediate_size"]), BR.Text(model / "tokenizer.json")))]
    out = {"context_mib": ctx}
    for name, build in builds:
        t = time.time()
        c = build()
        prog = elf.program(DATA / f"{prog_name}.elf")
        core.snapshot(c, ref.Machine.from_program(prog))
        toks = ref.Machine.from_program(prog).run(10 ** 6)
        build_s = time.time() - t
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        a0, u0 = torch.cuda.memory_allocated(), used()
        ex = Executor2(c, 1, "cuda")
        ex.load_init()
        torch.cuda.synchronize()
        weights = (torch.cuda.memory_allocated() - a0) / MiB
        proc = used() - u0
        del ex
        r = {"dims": len(c.dims), "build_s": round(build_s, 1), "executor_alloc_mib": round(weights, 1),
             "executor_process_growth_mib": round(proc, 1)}
        try:
            t = time.time()
            r["exact_positions"] = check(c, toks)
            r["check_s"] = round(time.time() - t, 1)
        except AssertionError as e:
            r["exact"] = f"MISMATCH {str(e)[:300]}"
        try:
            steps = min(2000, len(toks) - 25)
            sps = timed2(c, toks, steps=steps)
            r["seconds_per_step"] = sps
            r["tokens_per_s"] = round(1 / sps)
        except AssertionError as e:
            r["timing"] = f"MISMATCH {str(e)[:300]}"
        r["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / MiB, 1)
        out[name] = r
        print(name, json.dumps(r), flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
