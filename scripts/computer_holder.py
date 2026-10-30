"""Run B's companion: the Qwenputer computer (Triton Executor2) as its own process on the V100, computing non-stop,
printing its speed every ~2 s. Checked exact first (upstream's timed2 compares graph replays with the reference).

usage (in qwenputer/experimental/riscv): python computer_holder.py MODEL_DIR LOG [idle]
With "idle", it builds and checks the computer, then just holds its memory without computing.
"""
import json, sys, time
from pathlib import Path

import torch

from chipformer.q35.ir import Target
from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, ref
from chipformer.rv.fastexec import timed2
from chipformer.rv.measure import DATA
from chipformer.rv.tritonexec import Executor2

model, log = Path(sys.argv[1]), open(sys.argv[2], "a", buffering=1)
cfg = json.loads((model / "config.json").read_text())
tc = cfg.get("text_config", cfg)
c = core.build(Target(tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["intermediate_size"]),
               BR.Text(model / "tokenizer.json"))
prog = elf.program(DATA / "c-crc32.elf")
core.snapshot(c, ref.Machine.from_program(prog))
toks = ref.Machine.from_program(prog).run(10 ** 6)
sps = timed2(c, toks, steps=1000)              # asserts graph replays == reference on the first 300 steps
log.write(f"{time.strftime('%X')} exact check passed; {1 / sps:.0f} steps/s alone\n")
ex = Executor2(c, 1, "cuda"); ex.load_init()
emb = ex.embed(toks[:2000]); xin = torch.zeros(1, ex.D, device="cuda")
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        xin.copy_(emb[0:1]); ex.step(xin)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    ex.step(xin)
log.write(f"{time.strftime('%X')} ready\n")
if sys.argv[3:4] == ["idle"]:
    log.write(f"{time.strftime('%X')} holding memory, idle\n")
    while True:
        time.sleep(60)
n, t0 = 0, time.time()
while True:
    for t in range(len(emb)):
        xin.copy_(emb[t:t + 1]); g.replay(); n += 1
        if n % 500 == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            if dt >= 2:
                log.write(f"{time.strftime('%X')} {n / dt:.0f} steps/s\n"); n, t0 = 0, time.time()
