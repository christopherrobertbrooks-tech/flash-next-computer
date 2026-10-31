"""Speed probe for Qwenputer's computer on the V100 (SPEED-PLAN.md measurements 1-3). No language model.

For calculator traces (calc.elf on an expression, the RV32 core's own trace tokens):
  1. upstream's fastexec.timed2: the circuit's GPU time per step under one CUDA graph, at batch 1, 4, 16, 64;
  2. the same graph replayed per token with a copy of the output back to the CPU each token (today's round trip);
  3. a teacher-forced chunked prefill of the whole trace through the circuit as bf16 HF modules (upstream's
     gpucheck path: the snapshot injected into the cache, one forward call), timed, and checked exact on every IR dim at
     every position with upstream's hf.check (measurement only; the check itself runs on the CPU and is not timed).

usage (riscv dir on PYTHONPATH, the qwenputer venv, V100 visible, inside the V100 lock): python speed_probe.py OUT.json
"""
import json, sys, time

import torch

from chipformer.rv import calc, core, hf
from chipformer.rv.fastexec import timed2
from chipformer.rv.tritonexec import Executor2

EXPRS = ["123456789*987654321", "(2**64 - 1) // 7", "4/(-203)*962-40*31-(-37)*(-4)"]


def sync():
    torch.cuda.synchronize()


def round_trip(c, toks, n=1000):
    """Seconds per token for graph replay + copying the output back to the CPU every token (batch 1)."""
    ex = Executor2(c, 1, "cuda")
    ex.load_init()
    emb = ex.embed(toks[:n])
    x = torch.zeros(1, ex.D, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            x.copy_(emb[0:1])
            ex.step(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ex.step(x)
    sync()
    t = time.time()
    for i in range(len(emb)):
        x.copy_(emb[i:i + 1])
        g.replay()
        _ = x[0, :64].tolist()                       # the per-token device->host read the decode needs
    sync()
    return (time.time() - t) / len(emb)


def prefill(c, toks, reps=3):
    """Seconds for one chunked prefill of the whole trace through the circuit as bf16 HF modules."""
    model, lay, ids = hf.build_model(c, toks, torch.bfloat16)
    model = model.to("cuda")
    x = torch.tensor([[ids[t] for t in toks]], device="cuda")
    best = None
    with torch.no_grad():
        for _ in range(reps):
            out = model(input_ids=torch.tensor([[ids[None]]], device="cuda"), use_cache=True)
            cache = out.past_key_values
            hf.inject(c, cache)
            sync()
            t = time.time()
            model(input_ids=x, past_key_values=cache, use_cache=True)
            sync()
            dt = time.time() - t
            best = dt if best is None else min(best, dt)
    exact = hf.check(c, model, lay, ids, toks, "prefill")   # asserts on any inexact dim; returns positions
    del model
    torch.cuda.empty_cache()
    return best, exact


def main():
    out_path = sys.argv[1]
    c = core.build()
    results = {"device": torch.cuda.get_device_name(), "dims": len(c.dims), "layers": c.n_layers(), "exprs": []}
    for expr in EXPRS:
        core.snapshot(c, calc.machine(expr))
        toks = calc.machine(expr).run(10**6)
        r = {"expr": expr, "trace_tokens": len(toks), "timed2": {}}
        for b in (1, 4, 16, 64):
            s = timed2(c, toks, steps=min(500, len(toks) - 25), batch=b)
            r["timed2"][b] = {"s_per_step": s, "tokens_per_s": round(b / s)}
            print(f"{expr[:24]:24} timed2 batch {b:2}: {s * 1e6:7.1f} us/step, {b / s:9.0f} tokens/s", flush=True)
        rt = round_trip(c, toks)
        r["round_trip"] = {"s_per_token": rt, "tokens_per_s": round(1 / rt)}
        print(f"{expr[:24]:24} graph + CPU read per token: {rt * 1e6:7.1f} us, {1 / rt:7.0f} tokens/s", flush=True)
        try:
            pf, exact = prefill(c, toks)
            r["prefill"] = {"seconds": pf, "tokens_per_s": round(len(toks) / pf), "exact_positions": exact}
            print(f"{expr[:24]:24} prefill {len(toks)} tokens: {pf * 1000:.1f} ms, {len(toks) / pf:,.0f} tokens/s, "
                  f"exact on {exact} positions", flush=True)
        except AssertionError as e:
            r["prefill"] = {"error": "not exact: " + str(e)[:300]}
            print(f"{expr[:24]:24} prefill NOT EXACT: {str(e)[:200]}", flush=True)
        results["exprs"].append(r)
    json.dump(results, open(out_path, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
