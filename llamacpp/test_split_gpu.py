"""LLAMACPP-PLAN v2 stage 2 GPU controls on the stage-0 fixtures (V100, inside the lock). Exit 1 if any fails.

For each of the 12 fixtures:
1. exec_prompt_rows equals the HF prefill rows the fixture captured (bf16, elementwise);
2. the executor's head states and lag history after exec_prompt_rows equal those after upstream's own
   ExecRow.start on the same prompt and snapshot (bitwise);
3. a free-running split_generate (lock on): it must end EOS with no lock error; every row sent to the helper equals
   the fixture's row (the prefill, then each text step) for as long as the generated tokens agree with the fixture's;
   the first divergence is reported (position and HF margin there). Divergence itself isn't a failure: stage 0 showed
   99.78% teacher-forced agreement, so a free run can drift at a near-tie.
4. Reset: the first fixture is run again at the end and must give the same tokens (A, B, ..., A).

usage: python test_split_gpu.py MODEL_DIR GGUF HELPER FIXTURE_DIR
"""
import functools, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import split_chat as SC  # noqa: E402
from helper_client import Helper  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import chat as CH  # noqa: E402
from chipformer.rv import core, elf, ref  # noqa: E402

FIRMWARE = Path(CH.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"


class Rec:
    def __init__(self, h):
        self.h, self.sent = h, []

    def reset(self):
        self.sent = []
        return self.h.reset()

    def rows(self, pos, rows):
        self.sent.append(np.array(rows, copy=True))
        return self.h.rows(pos, rows)


def parked():
    m = ref.Machine.from_program(elf.program(FIRMWARE))
    BR.park(m)
    return m


def heads_state(chat):
    return [(st["S"].clone(), st["hist"].clone()) for st in chat.erow.heads]


def main():
    model, gguf, helper, fdir = sys.argv[1:5]
    fdir = Path(fdir)
    man = json.loads((fdir / "manifest.json").read_text())
    chat = SC.SplitChat(model)
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    def run(fx, h):
        ids = fx["prompt_ids"]
        lock = BR.Session(chat.T, parked())
        lock.history(ids)
        lock.qwen = functools.partial(lock.qwen, max_records=man["max_chip"])
        rec = Rec(h)
        r = SC.split_generate(chat, rec, ids, parked(), man["max_new"], max_chip=man["max_chip"], lock=lock)
        return r, rec

    first = None
    with Helper(helper, gguf, 16384, n_vocab=chat.v0) as h:
        for f in man["fixtures"]:
            fx = torch.load(fdir / f"{f['id']}.pt", weights_only=False)
            ids = fx["prompt_ids"]
            core.snapshot(chat.c, parked())
            rows = SC.exec_prompt_rows(chat, ids)
            s1 = heads_state(chat)
            chat._exec_start(ids)                                   # upstream's ExecRow.start
            s2 = heads_state(chat)
            same_state = all(torch.equal(a, b) and torch.equal(c, d) for (a, c), (b, d) in zip(s1, s2))
            check(f"{f['id']}: prompt rows = HF prefill rows ({rows.shape[0]}), executor state = ExecRow.start's",
                  torch.equal(rows.cpu(), fx["prefill_rows"]) and same_state)
            r, rec = run(fx, h)
            gen, ref_gen = r["tokens"], fx["generated"]
            d = next((i for i, (a, b) in enumerate(zip(gen, ref_gen)) if a != b), min(len(gen), len(ref_gen)))
            n_same = sum(1 for s in fx["steps"] if s["gen_index"] < d)        # text steps before the divergence
            pre_ok = np.array_equal(rec.sent[0], fx["prefill_rows"].float().numpy())
            steps_ok = all(np.array_equal(rec.sent[1 + k][0], fx["step_rows"][k].float().numpy())
                           for k in range(n_same))
            where = "identical to the fixture" if gen == ref_gen else f"first divergence at generated token {d}"
            if gen != ref_gen and d < len(ref_gen):
                st = [s for s in fx["steps"] if s["gen_index"] == d - 1]
                top = st[0]["top5"] if st else fx["prefill_top5"]
                where += f" (HF margin {top[1][0] - top[1][1]:.3f}, chip owner {st[0]['circuit'] if st else None})"
            ended = bool(gen) and gen[-1] in chat.eos
            check(f"{f['id']}: EOS {ended}, lock error {r['lock_error']}, {len(rec.sent) - 1} text rows, the first "
                  f"{n_same} equal the fixture's; {where}", ended and r["lock_error"] is None and pre_ok and steps_ok)
            if first is None:
                first = (fx, gen)
        r, _ = run(first[0], h)
        check(f"reset: {first[0]['id']} again after all others gives the same tokens", r["tokens"] == first[1])
    print("SPLIT GPU CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
