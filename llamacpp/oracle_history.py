"""Stage 2 history oracle, part 2 (V100, no helper): upstream's own HF chip-layer path (RvChat fast view: the
checkpoint's bf16 computer layers as one CUDA graph per token, the path upstream's executor "verify" mode checks
against) teacher-forced along each recorded split history. Saves, per question, the HF view's row after the chip
layers for every prompt position and every generated input (and the circuit's own choice there, circuit_token),
for compare_histories.py. Nothing here touches llama.cpp or the executor.

usage: python oracle_history.py MODEL_DIR HISTORY_DIR OUT_DIR   (in the V100 lock)
"""
import hashlib, json, os, sys
from pathlib import Path

import torch

from chipformer.rv import bridge as BR
from chipformer.rv import chat as CH
from chipformer.rv import core, elf, hf, ref

FIRMWARE = Path(CH.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"


@torch.no_grad()
def main():
    model, hdir, out = sys.argv[1:4]
    hdir, out = Path(hdir), Path(out)
    os.makedirs(out, exist_ok=False)
    chat = CH.RvChat(model, executor=False)
    assert chat.fast, "the HF fast chip view is needed (cuda)"
    man = json.loads((hdir / "manifest.json").read_text())
    files = {}
    for qid in man["files"]:
        hs = torch.load(hdir / f"{qid}.pt", weights_only=False)
        ids, gen = hs["prompt_ids"], hs["generated"]
        m = ref.Machine.from_program(elf.program(FIRMWARE))
        BR.park(m)
        core.snapshot(chat.c, m)                           # as generate(): the snapshot, then the prompt
        ccache = chat._chip_cache()
        x = torch.tensor([ids], device=chat.dev)
        h = chat.cm.embed_tokens(x)
        for L in range(chat.n_chip):
            h = chat.cm.layers[L](h, position_embeddings=None, attention_mask=None, position_ids=None,
                                  past_key_values=ccache)
        prompt_rows = h[0].to("cpu", copy=True)
        hf.inject(chat.c, ccache)
        if chat.graph is None:
            chat._capture()
        g = len(ids)
        rows, owner = [], []
        for t in gen[:-1] if gen and gen[-1] in chat.eos else gen:
            hc = chat._chip_step(t, g, ccache)
            g += 1
            owner.append(chat.circuit_token(hc))
            if t < chat.v0:
                rows.append(hc.to("cpu", copy=True))
        res = {"id": qid, "prompt_rows": prompt_rows, "step_rows": torch.stack(rows) if rows else None,
               "owner": owner}
        p = out / f"{qid}.pt"
        torch.save(res, p)
        files[qid] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{qid:14} {len(gen)} tokens through the HF chip view, {len(rows)} text rows", flush=True)
    (out / "manifest.json").write_text(json.dumps({"history": str(hdir), "files": files}, indent=1) + "\n")


if __name__ == "__main__":
    main()
