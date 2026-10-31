"""Stage 2 pipe-cost check: time qwen_half_helper's row steps (Python client, pipe, llama.cpp decode) on a fixture's
real rows, teacher-forced: the prompt as one prefill, then N single-row steps. Compare the per-step time with
llama-bench's tg time on the same GGUF (llama.cpp's decode alone, token input) to get the pipe's share.

usage: python bench_helper.py HELPER GGUF FIXTURE.pt [N]   (V100 visible, in the lock)
"""
import statistics, sys, time

import torch

sys.path.insert(0, __import__("os").path.dirname(__file__))
from helper_client import Helper  # noqa: E402


def main():
    helper, gguf, fxp = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    fx = torch.load(fxp, weights_only=False)
    P = len(fx["prompt_ids"])
    with Helper(helper, gguf, 16384, n_vocab=248320) as h:
        for rep in range(2):                              # the first pass warms up (graphs, caches)
            h.reset()
            t = time.perf_counter()
            h.rows(list(range(P)), fx["prefill_rows"].float().numpy())
            pre = time.perf_counter() - t
            ts = []
            for k in range(min(n, fx["step_rows"].shape[0])):
                t = time.perf_counter()
                h.rows([P + k], fx["step_rows"][k:k + 1].float().numpy())
                ts.append(time.perf_counter() - t)
        print(f"{gguf}: prefill {P} rows {pre * 1000:.1f} ms ({P / pre:.0f} rows/s); {len(ts)} steps: median "
              f"{statistics.median(ts) * 1000:.2f} ms, mean {statistics.mean(ts) * 1000:.2f} ms "
              f"({1 / statistics.median(ts):.1f} steps/s)")


if __name__ == "__main__":
    main()
