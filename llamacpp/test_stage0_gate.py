"""Negative controls for stage0_gate.py (Codex stage-0 review finding 2): each broken copy must FAIL or STOP, and the
real data must PASS. Works in a scratch folder (unchanged fixtures are symlinked); never touches the real files.

usage: python test_stage0_gate.py FIXTURE_DIR SPLIT.json PLAIN_HF_V2.json PLAIN_LLAMA.json PLAIN_HF_V1.json SCRATCH
"""
import copy, hashlib, json, os, shutil, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import stage0_gate as G  # noqa: E402


def main():
    fdir, sp, hp, lp, h1, scratch = map(Path, sys.argv[1:7])
    load = lambda p: json.loads(p.read_text())
    split, phf, pll, phf1 = load(sp), load(hp), load(lp), load(h1)
    man0 = load(fdir / "manifest.json")
    bad = 0

    def setup(man, replace=None):
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        for f in man0["fixtures"]:
            if not replace or f["id"] not in replace:
                os.symlink((fdir / f"{f['id']}.pt").resolve(), scratch / f"{f['id']}.pt")
        for k, fx in (replace or {}).items():
            torch.save(fx, scratch / f"{k}.pt")
        (scratch / "manifest.json").write_text(json.dumps(man))
        return scratch

    def run(label, want, man, s=split, h=phf, l=pll, replace=None):
        nonlocal bad
        d = setup(man, replace)
        try:
            *_, ok = G.gate(d, s, h, l)
            got = "PASS" if ok else "STOP"
        except G.Fail as e:
            got = f"FAIL ({e})"
        ok_ = got.startswith(want)
        bad += not ok_
        print(f"{'ok ' if ok_ else 'BAD'} {label}: {got[:110]}", flush=True)

    run("real data", "PASS", man0)
    fx = torch.load(fdir / "final-2-001.pt", weights_only=False)
    fx_lock = dict(fx, stop="lock error", lock_error={"at": 5, "got": "X", "want": "Y"})
    man = copy.deepcopy(man0)
    d = setup(man, {"final-2-001": fx_lock})
    h = hashlib.sha256((d / "final-2-001.pt").read_bytes()).hexdigest()
    for f in man["fixtures"]:
        if f["id"] == "final-2-001":
            f["sha256"] = h                               # a consistent hash: only the lock check can catch it
    run("a fixture with a lock error (hash updated)", "FAIL", man, replace={"final-2-001": fx_lock})
    man = copy.deepcopy(man0)
    man["fixtures"][3]["sha256"] = "0" * 64
    run("a manifest hash that doesn't match its file", "FAIL", man)
    man = copy.deepcopy(man0)
    man["fixtures"] = man["fixtures"][:1]
    run("a manifest with 1 of the 12 questions", "FAIL", man)
    man = copy.deepcopy(man0)
    man["fixtures"].append(man["fixtures"][0])
    run("a duplicated question", "FAIL", man)
    s = copy.deepcopy(split)
    s["_reset_check"]["same"] = False
    run("the split reset check failed", "FAIL", man0, s=s)
    run("a plain HF reference without recorded argmax (v1)", "FAIL", man0, h=phf1)
    s = copy.deepcopy(split)
    for qid in ("final-1-001", "final-2-001", "final-3-001"):
        fxq = torch.load(fdir / f"{qid}.pt", weights_only=False)
        top = [fxq["prefill_top5"]] + [st["top5"] for st in fxq["steps"]]
        qpos = [0] + [k + 1 for k, st in enumerate(fxq["steps"]) if st["circuit"] is None]
        j = max(qpos, key=lambda j: top[j][1][0] - top[j][1][1])
        s[qid][j][0][0] = (s[qid][j][0][0] + 1) % man0["v0"]
    run("3 questions with a high-margin split disagreement", "STOP", man0, s=s)
    s = copy.deepcopy(split)                              # a helper that answers EOS at every position (Codex stage 2)
    for qid in s:
        if not qid.startswith("_"):
            for e in s[qid]:
                e[0][0] = 248046
    run("a helper that always answers EOS", "STOP", man0, s=s)
    shutil.rmtree(scratch)
    print("GATE CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
