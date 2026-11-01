"""LLAMACPP-PLAN v2 stage 0d, llama.cpp side: drive qwen_half_helper over the fixtures and record the top 5 at every
Qwen position (the prompt's last, then each text step), exactly as the fixtures order them.

  --mode split: the Qwen-half GGUF fed the fixture rows (prefill rows at 0..P-1, then each step's row at its text
                position). Hard checks: each row converts bf16 -> f32 -> bf16 unchanged; the step positions are P, P+1,
                ... (chip tokens never advance them); after all fixtures the first one is run again and must give
                bitwise the same top 5 at every position (reset: A, B, ..., A).
  --mode plain: the plain Qwen3.5-4B f16 GGUF fed token ids (the prompt, then each step's input id): the control.

usage: python run_helper_fixtures.py --mode split|plain HELPER GGUF FIXTURE_DIR OUT.json   (V100 visible, in the lock)
"""
import argparse, json, struct, subprocess, sys
from pathlib import Path

import numpy as np
import torch


class Helper:
    def __init__(self, exe, gguf):
        self.p = subprocess.Popen([exe, gguf], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _reply(self, rows):
        st = struct.unpack("<i", self.p.stdout.read(4))[0]
        if st != 0:
            n = struct.unpack("<I", self.p.stdout.read(4))[0]
            raise RuntimeError("helper error: " + self.p.stdout.read(n).decode())
        if not rows:
            return None
        ids = list(struct.unpack("<5i", self.p.stdout.read(20)))
        vals = list(struct.unpack("<5f", self.p.stdout.read(20)))
        return [ids, vals]

    def reset(self):
        self.p.stdin.write(struct.pack("<II", 1, 0)); self.p.stdin.flush()
        return self._reply(False)

    def rows(self, pos, rows):
        rows = np.ascontiguousarray(rows, dtype="<f4")
        self.p.stdin.write(struct.pack("<II", 2, len(pos)) + np.asarray(pos, "<i4").tobytes() + rows.tobytes())
        self.p.stdin.flush()
        return self._reply(True)

    def tokens(self, pos, ids):
        self.p.stdin.write(struct.pack("<II", 4, len(pos)) + np.asarray(pos, "<i4").tobytes()
                           + np.asarray(ids, "<i4").tobytes())
        self.p.stdin.flush()
        return self._reply(True)

    def close(self):
        try:
            self.p.stdin.write(struct.pack("<II", 3, 0)); self.p.stdin.flush()
            self.p.wait(30)
        finally:
            if self.p.poll() is None:
                self.p.kill()
        if self.p.returncode != 0:
            raise RuntimeError(f"helper exit {self.p.returncode}")


def f32(t):
    x = t.float()
    assert torch.equal(x.to(t.dtype), t), "a row didn't survive bf16 -> f32 -> bf16"
    return x.numpy()


def run_split(h, fx):
    P = len(fx["prompt_ids"])
    h.reset()
    tops = [h.rows(list(range(P)), f32(fx["prefill_rows"]))]
    rows = fx["step_rows"]
    for k, s in enumerate(fx["steps"]):
        assert s["pos"] == P + k, (fx["id"], k, s["pos"], P + k)
        tops.append(h.rows([s["pos"]], f32(rows[k:k + 1])))
    return tops


def run_plain(h, fx):
    P = len(fx["prompt_ids"])
    h.reset()
    tops = [h.tokens(list(range(P)), fx["prompt_ids"])]
    for k, s in enumerate(fx["steps"]):
        tops.append(h.tokens([P + k], [s["input_id"]]))
    return tops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["split", "plain"], required=True)
    ap.add_argument("helper"); ap.add_argument("gguf"); ap.add_argument("fixtures"); ap.add_argument("out")
    a = ap.parse_args()
    d = Path(a.fixtures)
    man = json.loads((d / "manifest.json").read_text())
    h = Helper(a.helper, a.gguf)
    res, first = {}, None
    try:
        for f in man["fixtures"]:
            fx = torch.load(d / f"{f['id']}.pt", weights_only=False)
            res[f["id"]] = (run_split if a.mode == "split" else run_plain)(h, fx)
            first = first or fx
            print(f["id"], len(res[f["id"]]), flush=True)
        again = (run_split if a.mode == "split" else run_plain)(h, first)
        res["_reset_check"] = {"id": first["id"], "same": again == res[first["id"]]}
        print("reset check (A, B, ..., A):", res["_reset_check"], flush=True)
    finally:
        h.close()
    Path(a.out).write_text(json.dumps(res) + "\n")
    if not res["_reset_check"]["same"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
