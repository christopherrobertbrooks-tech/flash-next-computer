"""Fused-executor premise check: where does the computer's time per chip token go in arm F's loop?

1. **The real loop:** split_generate in token mode (arm F's GGUF) on a few questions, with the executor step wrapped:
   CUDA events around each graph replay (its GPU time), and wall time around the whole chip step (the embedding
   lookup and copy, the replay, code_token's decode, which syncs). Compared with the loop's own chip bucket.
2. **The GPU ceiling:** the same graph replayed N times back to back (no host work between): GPU time per step with
   no Python gaps, i.e. what a device-resident decode loop could at best reach with today's kernels.
3. **Kernels vs gaps:** torch.profiler over M replays: the sum of kernel durations per step, against (2); the
   difference is launch/scheduling gaps inside the graph, the most a fusion of its 36 launches could remove.
Measurement only; nothing is changed. Prints a JSON summary.

usage: python profile_chip_step.py MODEL_DIR GGUF HELPER QUESTIONS.jsonl ID[,ID...] OUT.json   (V100, in the lock)
"""
import functools, json, statistics, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import split_chat as SC  # noqa: E402
from helper_client import Helper  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import chat as CH  # noqa: E402
from chipformer.rv import elf, ref  # noqa: E402

FIRMWARE = Path(CH.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"


def parked():
    m = ref.Machine.from_program(elf.program(FIRMWARE))
    BR.park(m)
    return m


def main():
    model, gguf, helper, qpath, ids, out = sys.argv[1:7]
    qs = {json.loads(l)["id"]: json.loads(l) for l in open(qpath)}
    chat = SC.SplitChat(model)
    er = chat.erow
    rec = {"step_wall": [], "events": [], "code_wall": []}
    real_step, real_code = chat._exec_step, chat.code_token

    def timed_step(t):
        w0 = time.perf_counter()
        er.xbuf.copy_(chat._exec_emb([t]))
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        er.graph.replay()
        b.record()
        rec["events"].append((a, b))
        rec["_w0"] = w0
        return er.kbuf

    def timed_code(code):
        c0 = time.perf_counter()
        r = real_code(code)
        c1 = time.perf_counter()
        rec["code_wall"].append(c1 - c0)
        rec["step_wall"].append(c1 - rec["_w0"])
        return r

    summary = {"questions": []}
    with Helper(helper, gguf, 16384, n_vocab=chat.v0) as h:
        for qid in ids.split(","):
            ids_ = chat.prompt_ids([{"role": "user", "content": qs[qid]["prompt"]}])
            lock = BR.Session(chat.T, parked())
            lock.history(ids_)
            lock.qwen = functools.partial(lock.qwen, max_records=2_000_000)
            rec.update(step_wall=[], events=[], code_wall=[])
            chat._exec_step, chat.code_token = timed_step, timed_code
            t0 = time.perf_counter()
            r = SC.split_generate(chat, h, ids_, parked(), 8192, max_chip=2_000_000, lock=lock, lm_input="tokens")
            wall = time.perf_counter() - t0
            chat._exec_step, chat.code_token = real_step, real_code
            torch.cuda.synchronize()
            gpu = [a.elapsed_time(b) / 1e3 for a, b in rec["events"]]
            n = r["chip_tokens"]
            # text steps also run an executor step (no code_token there); keep the chip ones: those followed by code
            chip_gpu = gpu[:len(gpu)]
            summary["questions"].append({
                "id": qid, "chip_tokens": n, "text_tokens": r["text_tokens"], "lock_error": r["lock_error"],
                "wall_s": round(wall, 2), "chip_bucket_s": r["chip_seconds"],
                "chip_bucket_us_per_token": round(1e6 * r["chip_seconds"] / max(1, n), 1),
                "executor_steps": len(gpu),
                "replay_gpu_us_median": round(1e6 * statistics.median(gpu), 1),
                "replay_gpu_us_mean": round(1e6 * statistics.mean(gpu), 1),
                "chip_step_wall_us_median": round(1e6 * statistics.median(rec["step_wall"]), 1) if rec["step_wall"] else None,
                "code_token_wall_us_median": round(1e6 * statistics.median(rec["code_wall"]), 1) if rec["code_wall"] else None})
            print(json.dumps(summary["questions"][-1]), flush=True)
    # 2: back-to-back replays (the GPU ceiling with today's kernels)
    N = 2000
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(N):
        er.graph.replay()
    b.record()
    torch.cuda.synchronize()
    b2b = a.elapsed_time(b) / 1e3 / N
    # 3: kernel durations inside the graph
    from torch.profiler import ProfilerActivity, profile
    M = 50
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(M):
            er.graph.replay()
        torch.cuda.synchronize()
    kern = {}
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" and e.self_device_time_total > 0:
            kern[e.key] = {"calls_per_step": e.count / M, "us_per_step": e.self_device_time_total / M}
    ksum = sum(v["us_per_step"] for v in kern.values())
    summary["back_to_back_us_per_step"] = round(1e6 * b2b, 1)
    summary["kernels_us_per_step"] = round(ksum, 1)
    summary["kernel_launches_per_step"] = round(sum(v["calls_per_step"] for v in kern.values()), 1)
    summary["kernels"] = {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in
                          sorted(kern.items(), key=lambda kv: -kv[1]["us_per_step"])[:12]}
    summary["gaps_us_per_step"] = round(1e6 * b2b - ksum, 1)
    Path(out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "kernels"}, indent=1))
    print(json.dumps(summary["kernels"], indent=1))


if __name__ == "__main__":
    main()
