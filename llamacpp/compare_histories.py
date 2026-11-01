"""Stage 2 history oracle, part 3 (CPU): compare the split's recorded rows with the HF chip view's rows along the
same history, at every position (the prompt and every text step, before and after any divergence from the old
fixture), and the circuit's ownership: wherever the HF view's circuit owns the next token, the split must have
generated exactly that token. It first checks both manifests' file hashes and each record's structure (ids, row,
position and ownership counts tied to the tokens, a clean EOS history). Also runs its own negative controls on the
same data (empty text positions, a lock error, a corrupted suffix row, a corrupted prompt row, a changed
circuit-owned token) and requires each to be caught. Exit 1 on any mismatch or a
control that isn't caught.

usage: python compare_histories.py HISTORY_DIR ORACLE_DIR
"""
import copy, json, sys
from pathlib import Path

import torch


def structure(hs, orc, v0=248320, eos=(248046, 248044)):
    """Problems with the records' shape (Codex confirm, finding 7): ids, counts tied to the tokens, positions,
    ownership length, EOS/lock status."""
    probs = []
    gen, P = hs["generated"], len(hs["prompt_ids"])
    if hs["id"] != orc["id"]:
        probs.append(f"ids differ ({hs['id']} vs {orc['id']})")
    if hs["lock_error"] is not None or not gen or gen[-1] not in eos:
        probs.append(f"not a clean EOS history (lock {hs['lock_error']}, last {gen[-1] if gen else None})")
    fed = gen[:-1] if gen and gen[-1] in eos else gen
    n_text = sum(t < v0 for t in fed)
    n_rows = 0 if hs["step_rows"] is None else hs["step_rows"].shape[0]
    if hs["prefill_rows"].shape[0] != P or hs["prefill_pos"] != list(range(P)):
        probs.append("the prefill isn't the prompt at 0..P-1")
    if n_rows != n_text or len(hs["step_pos"]) != n_text:
        probs.append(f"{n_rows} text rows / {len(hs['step_pos'])} positions for {n_text} text inputs")
    if len(orc["owner"]) != len(fed):
        probs.append(f"{len(orc['owner'])} ownership entries for {len(fed)} inputs")
    return probs


def compare(hs, orc, v0=248320):
    """Returns a list of problems (empty = every row and every circuit choice agrees)."""
    probs = structure(hs, orc, v0)
    if not torch.equal(hs["prefill_rows"], orc["prompt_rows"].to(hs["prefill_rows"].dtype)):
        probs.append("prompt rows differ")
    a, b = hs["step_rows"], orc["step_rows"]
    if (a is None) != (b is None) or (a is not None and a.shape != b.shape):
        probs.append(f"text row counts differ ({None if a is None else a.shape[0]} vs {None if b is None else b.shape[0]})")
    elif a is not None:
        bad = (a != b.to(a.dtype)).any(1).nonzero().flatten().tolist()
        if bad:
            probs.append(f"{len(bad)} text rows differ (first {bad[:5]})")
    P = len(hs["prompt_ids"])
    if hs["step_pos"] != list(range(P, P + len(hs["step_pos"]))):
        probs.append("text positions aren't P, P+1, ...")
    gen = hs["generated"]
    for i, ct in enumerate(orc["owner"]):
        if ct is not None and i + 1 < len(gen) and gen[i + 1] != ct:
            probs.append(f"generated token {i + 1} is {gen[i + 1]}, the circuit owns {ct}")
            break
    return probs


def main():
    hdir, odir = Path(sys.argv[1]), Path(sys.argv[2])
    man = json.loads((hdir / "manifest.json").read_text())
    oman = json.loads((odir / "manifest.json").read_text())
    bad = 0
    import hashlib
    for d, m in ((hdir, man), (odir, oman)):             # every file must be the one its manifest names
        for qid, h in m["files"].items():
            if hashlib.sha256((d / f"{qid}.pt").read_bytes()).hexdigest() != h:
                print(f"BAD {d / qid}.pt doesn't match its manifest hash")
                bad += 1
    if set(man["files"]) != set(oman["files"]):
        print("BAD the history and oracle manifests list different questions")
        bad += 1
    tot_rows = tot_owned = 0
    first = None
    for qid in man["files"]:
        hs = torch.load(hdir / f"{qid}.pt", weights_only=False)
        orc = torch.load(odir / f"{qid}.pt", weights_only=False)
        probs = compare(hs, orc)
        n = 0 if hs["step_rows"] is None else hs["step_rows"].shape[0]
        owned = sum(c is not None for c in orc["owner"])
        tot_rows += n + hs["prefill_rows"].shape[0]
        tot_owned += owned
        bad += bool(probs)
        print(f"{'ok ' if not probs else 'BAD'} {qid:14} {hs['prefill_rows'].shape[0]} prompt + {n} text rows equal, "
              f"{owned} circuit-owned choices agree{'' if not probs else ': ' + '; '.join(probs)}", flush=True)
        if first is None and n > 10:
            first = (hs, orc)
    hs, orc = first
    for label, mut in (("empty text positions", lambda h: h.__setitem__("step_pos", [])),
                       ("a lock error in the history", lambda h: h.__setitem__("lock_error", {"at": 1})),("a corrupted suffix row", lambda h: h["step_rows"][-1].__setitem__(5, h["step_rows"][-1][5] + 1)),
                       ("a corrupted prompt row", lambda h: h["prefill_rows"][3].__setitem__(100, 1.0)),
                       ("a changed circuit-owned token",
                        lambda h: h["generated"].__setitem__(1 + next(i for i, c in enumerate(orc["owner"]) if c is not None),
                                                             -5))):
        h2 = copy.deepcopy(hs)
        mut(h2)
        caught = bool(compare(h2, orc))
        bad += not caught
        print(f"{'ok ' if caught else 'BAD'} control: {label} is caught", flush=True)
    print(f"HISTORY ORACLE {'PASSED' if not bad else f'FAILED ({bad})'}: {tot_rows} rows and {tot_owned} circuit "
          f"choices compared")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
