"""Stage 2 history oracle, part 1 (V100, helper): run split_generate (lock on) on the stage-0 fixture prompts and
save, per question, the exact history: the generated ids and every row sent to the helper (the prompt's prefill
rows, then one row per text step, as bf16, the dtype they were built in). part 2 (oracle_history.py) recomputes the
rows on upstream's independent HF chip-layer path along this same history.

usage: python record_split_history.py MODEL_DIR GGUF HELPER FIXTURE_DIR OUT_DIR   (in the V100 lock)
"""
import functools, hashlib, json, os, sys
from pathlib import Path

import numpy as np
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


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Rec:
    def __init__(self, h):
        self.h, self.sent = h, []

    def reset(self):
        self.sent = []
        return self.h.reset()

    def rows(self, pos, rows):
        r = torch.from_numpy(np.array(rows, copy=True))
        assert torch.equal(r.to(torch.bfloat16).float(), r), "a row isn't exactly bf16"
        self.sent.append((list(pos), r.to(torch.bfloat16)))
        return self.h.rows(pos, rows)


def main():
    model, gguf, helper, fdir, out = sys.argv[1:6]
    fdir, out = Path(fdir), Path(out)
    os.makedirs(out, exist_ok=False)
    man = json.loads((fdir / "manifest.json").read_text())
    chat = SC.SplitChat(model)
    files = {}
    gguf_sha, helper_sha = _sha(gguf), _sha(helper)       # whole-file hashes (the first version hashed 1 MiB)
    with Helper(helper, gguf, 16384, n_vocab=chat.v0) as h:
        for f in man["fixtures"]:
            fx = torch.load(fdir / f"{f['id']}.pt", weights_only=False)
            ids = fx["prompt_ids"]
            lock = BR.Session(chat.T, parked())
            lock.history(ids)
            lock.qwen = functools.partial(lock.qwen, max_records=man["max_chip"])
            rec = Rec(h)
            r = SC.split_generate(chat, rec, ids, parked(), man["max_new"], max_chip=man["max_chip"], lock=lock)
            pre_pos, pre = rec.sent[0]
            steps = rec.sent[1:]
            hist = {"id": f["id"], "prompt_ids": ids, "generated": r["tokens"], "lock_error": r["lock_error"],
                    "prefill_pos": pre_pos, "prefill_rows": pre, "step_pos": [p[0] for p, _ in steps],
                    "step_rows": torch.cat([x for _, x in steps]) if steps else None,
                    "gguf_sha256": gguf_sha, "helper_sha256": helper_sha}
            p = out / f"{f['id']}.pt"
            torch.save(hist, p)
            files[f["id"]] = hashlib.sha256(p.read_bytes()).hexdigest()
            print(f"{f['id']:14} {len(r['tokens'])} tokens, {len(steps)} text rows, lock error {r['lock_error']}",
                  flush=True)
    (out / "manifest.json").write_text(json.dumps({"gguf": gguf, "gguf_sha256": gguf_sha, "helper": helper,
                                                   "helper_sha256": helper_sha, "files": files}, indent=1) + "\n")


if __name__ == "__main__":
    main()
