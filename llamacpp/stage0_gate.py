"""LLAMACPP-PLAN v2 stage 0: the fail-closed gate (Codex stage-0 review findings 1-2). Exit 0 only if every hard check
passes AND the fixed stop rule passes; any failed check, missing or extra file, or bad identity exits 1 with the reason.

Hard checks:
- the manifest holds exactly the 12 fixed ids, once each (an extra question only if none of the 12 has two calls; it
  never counts in the rule), and each fixture file matches its manifest sha256;
- every fixture ended EOS with no lock error, under both budgets; prompt rows = prompt length; one step row per step;
  step positions P, P+1, ...; every step's input id = the generated id at its index; ownership recorded everywhere;
- the executor-built prompt rows equal the HF prefill rows (bf16, elementwise);
- the split / plain-llama / plain-HF outputs exist for exactly these ids with one entry per Qwen position, ids inside
  the text vocabulary, finite values; both llama runs' A, B, ..., A reset checks are true;
- optional: --provenance PROV.json; every file it lists must still match its sha256.
The rule (fixed in the plan): the greedy choice is the argmax (the lowest id at a tie). The fixtures' HF choice at a
Qwen-owned position is the token upstream actually generated next (it decides by torch.argmax). The plain HF choice
is the recorded argmax (plain_hf_reference v2; a reference without it is refused). The helper's ids[0] is the lowest
id at the maximum. STOP if split agreement < control - 1.0 point, or > 2 questions have a split disagreement at an HF
margin >= 1.0 logit.

usage: python stage0_gate.py FIXTURE_DIR SPLIT.json PLAIN_HF.json PLAIN_LLAMA.json [--provenance PROV.json]
"""
import argparse, hashlib, json, math, sys
from pathlib import Path

import torch

FIXED = [f"final-{g}-00{k}" for g in (1, 2, 3, 4) for k in (1, 2, 3)]


class Fail(Exception):
    pass


def need(cond, why):
    if not cond:
        raise Fail(why)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_output(name, out, ids, lengths, v0, needs_argmax=False):
    need(isinstance(out, dict), f"{name}: not a json object")
    got = {k for k in out if not k.startswith("_")}
    need(got == set(ids), f"{name}: ids {sorted(got ^ set(ids))} missing or extra")
    for i in ids:
        need(len(out[i]) == lengths[i], f"{name} {i}: {len(out[i])} entries, want {lengths[i]}")
        for e in out[i]:
            need(len(e[0]) == 5 and all(0 <= t < v0 for t in e[0]), f"{name} {i}: id out of range")
            need(all(math.isfinite(v) for v in e[1]), f"{name} {i}: non-finite value")
            if needs_argmax:
                need(len(e) == 3 and 0 <= e[2] < v0, f"{name} {i}: no recorded argmax (use plain_hf_reference v2)")


def gate(fdir, split, phf, pll, prov=None):
    man = json.loads((fdir / "manifest.json").read_text())
    ids = [f["id"] for f in man["fixtures"]]
    need(len(ids) == len(set(ids)), "duplicate fixture ids")
    fixed = [i for i in ids if not i.startswith("extra")]
    need(sorted(fixed) == sorted(FIXED), f"fixed ids differ: missing {sorted(set(FIXED) - set(fixed))}, "
                                        f"extra {sorted(set(fixed) - set(FIXED))}")
    v0 = man["v0"]
    if prov:
        for path, h in json.loads(Path(prov).read_text())["files"].items():
            need(Path(path).exists() and sha(path) == h, f"provenance: {path} changed or missing")
    fx_all, lengths, calls = {}, {}, {}
    for f in man["fixtures"]:
        p = fdir / f"{f['id']}.pt"
        need(p.exists(), f"{f['id']}: fixture file missing")
        need(sha(p) == f["sha256"], f"{f['id']}: fixture sha256 differs from the manifest")
        fx = torch.load(p, weights_only=False)
        need(fx["id"] == f["id"], f"{f['id']}: file holds {fx['id']}")
        need(fx["stop"] == "eos" and fx["lock_error"] is None, f"{f['id']}: stop {fx['stop']}, lock {fx['lock_error']}")
        need(fx["text_tokens"] < man["max_new"] and fx["chip_tokens"] < man["max_chip"], f"{f['id']}: at a budget")
        need(fx["generated"] and fx["generated"][-1] in (248046, 248044), f"{f['id']}: doesn't end with EOS")
        P = len(fx["prompt_ids"])
        need(tuple(fx["prefill_rows"].shape) == (P, 3968), f"{f['id']}: prefill rows {tuple(fx['prefill_rows'].shape)}")
        need(torch.equal(fx["executor_prompt_rows"].to(fx["prefill_rows"].dtype), fx["prefill_rows"]),
             f"{f['id']}: executor prompt rows differ from the HF prefill rows")
        st = fx["steps"]
        need(fx["step_rows"] is not None and fx["step_rows"].shape[0] == len(st), f"{f['id']}: step rows")
        for k, s in enumerate(st):
            need(s["pos"] == P + k, f"{f['id']}: step {k} at position {s['pos']}")
            need(fx["generated"][s["gen_index"]] == s["input_id"], f"{f['id']}: step {k} input id")
            need("circuit" in s, f"{f['id']}: step {k} has no ownership")
        need(sum(1 for s in st if s["input_id"] >= v0) == 0, f"{f['id']}: a chip token reached Qwen")
        fx_all[f["id"]], lengths[f["id"]] = fx, 1 + len(st)
        calls[f["id"]] = fx["calls"]
    extra = [i for i in ids if i.startswith("extra")]
    need(not extra or max(calls[i] for i in fixed) < 2, "an extra question although a fixed one has two calls")
    check_output("split", split, ids, lengths, v0)
    check_output("plain llama", pll, ids, lengths, v0)
    check_output("plain HF", phf, ids, lengths, v0, needs_argmax=True)
    for name, out in (("split", split), ("plain llama", pll)):
        need(out.get("_reset_check", {}).get("same") is True, f"{name}: the A, B, ..., A reset check didn't pass")

    S = C = N = real_q = 0
    rows = []
    for i in FIXED:
        fx = fx_all[i]
        hf_top = [fx["prefill_top5"]] + [s["top5"] for s in fx["steps"]]
        gen = fx["generated"]
        nxt = [gen[0]] + [gen[s["gen_index"] + 1] for s in fx["steps"]]   # what upstream chose after each position
        qpos = [0] + [k + 1 for k, s in enumerate(fx["steps"]) if s["circuit"] is None]
        for j in qpos:                                   # the chosen id must be one of the maximal logits
            ids5, vals5 = hf_top[j]
            need(nxt[j] not in ids5 or vals5[ids5.index(nxt[j])] == vals5[0], f"{i}: position {j} choice not maximal")
        s_dis = [j for j in qpos if split[i][j][0][0] != nxt[j]]
        c_dis = [j for j in qpos if pll[i][j][0][0] != phf[i][j][2]]
        margin = lambda j: hf_top[j][1][0] - hf_top[j][1][1]
        real = [j for j in s_dis if margin(j) >= 1.0]
        real_q += bool(real)
        S += len(qpos) - len(s_dis)
        C += len(qpos) - len(c_dis)
        N += len(qpos)
        rows.append(f"{i:14} {len(qpos):5} positions | split disagree {len(s_dis):3} (>= 1.0 logit: {len(real)}) "
                    f"| control disagree {len(c_dis):3}")
    sa, ca = 100 * S / N, 100 * C / N
    return rows, sa, ca, N, real_q, (sa >= ca - 1.0 and real_q <= 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixtures"); ap.add_argument("split"); ap.add_argument("plain_hf"); ap.add_argument("plain_llama")
    ap.add_argument("--provenance")
    a = ap.parse_args()
    try:
        load = lambda p: json.loads(Path(p).read_text())
        rows, sa, ca, N, real_q, ok = gate(Path(a.fixtures), load(a.split), load(a.plain_hf), load(a.plain_llama),
                                           a.provenance)
    except (Fail, FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"GATE FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
    print("\n".join(rows))
    print(f"hard checks: all passed. Over {N} Qwen-chosen positions (greedy argmax, lowest id at a tie): split "
          f"{sa:.3f}%, control {ca:.3f}% (rule: split >= control - 1.0); questions with a split disagreement at "
          f"margin >= 1.0: {real_q} (rule: <= 2)")
    print("PREMISE", "PASS" if ok else "STOP")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
