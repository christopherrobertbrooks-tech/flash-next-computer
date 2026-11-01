"""CPU regression checks for the scorer's arms F and G (FAST-PLAN; Codex fast review finding 4), on real records:
stage 3's 160 S replies relabeled as F and as G.
- F,C; G,C; F,S; G,S each score both arms (160 rows), name the comparison, give the signed F_minus_C (etc.) interval
  and the first-arm-relative latency rule;
- F vs S on identical records (relabeled) has no discordant pairs and a zero difference;
- an unknown arm, a comparison naming a missing arm, a duplicated id and a missing id are refused (nonzero exit, or
  the verdict withheld), never scored as a subset;
- stage 3 (S,C), stage 1 (L,R) and inside-4b (C,R) re-score identically to their published scores.

usage: python llamacpp/test_score_fast.py SCRATCH   (repo root)
"""
import gzip, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE, Q = ROOT / "stage5/score.py", ROOT / "inside4b/final-primary.jsonl"


def main():
    scr = Path(sys.argv[1])
    if scr.exists():
        shutil.rmtree(scr)
    scr.mkdir(parents=True)
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    def arm_dir(name, src, label=None, edit=None):
        d = scr / name
        d.mkdir()
        shutil.copy(ROOT / src / "questions.sha256", d)
        rows = [json.loads(l) for l in gzip.open(ROOT / src / "replies.jsonl.gz", "rt")]
        if label:
            rows = [dict(r, arm=label) for r in rows]
        if edit:
            rows = edit(rows)
        (d / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return d

    def score(out, *args):
        r = subprocess.run([sys.executable, str(SCORE), str(Q), str(scr / out), *map(str, args)], capture_output=True,
                           text=True)
        return r.returncode, (json.load(open(scr / out)) if (scr / out).exists() and r.returncode == 0 else None), \
            r.stdout + r.stderr

    st3 = "results/llamacpp/stage3"
    F, G = arm_dir("F", st3, "F"), arm_dir("G", st3, "G")
    C, S = arm_dir("C", "results/inside4b-full/full-C"), arm_dir("S", st3)
    dirs = {"F": F, "G": G, "C": C, "S": S}
    for a, b in (("F", "C"), ("G", "C"), ("F", "S"), ("G", "S")):
        rc, o, _ = score(f"{a}{b}.json", f"{a}={dirs[a]}", f"{b}={dirs[b]}", "--compare", f"{a},{b}")
        p = o["summary"]["primary"] if o else {}
        ok = rc == 0 and p.get("compared") == f"{a} vs {b}" and f"{a}_minus_{b}" in p \
            and sum(p["pairs"].values()) == 160 and set(p["median_seconds"]) == {a, b} and "ci95" in p
        if b == "S":                                       # identical records relabeled: nothing discordant
            ok = ok and p["pairs"][f"{a} only"] == 0 and p["pairs"]["S only"] == 0 and p[f"{a}_minus_S"] == 0
        if b == "C":                                       # the same records as S: S vs C's numbers
            ok = ok and p[f"{a}_minus_C"] == -0.025 and p["ci95"] == [-0.0877, 0.0371]
        check(f"--compare {a},{b}: {p.get('compared')}, pairs {p.get('pairs')}, {a}_minus_{b} {p.get(f'{a}_minus_{b}')}",
              ok)
    rc, _, out = score("u.json", f"F={F}", f"X={C}")
    check(f"an unknown arm is refused (exit {rc})", rc != 0 and "unknown arm" in out)
    rc, _, out = score("m.json", f"F={F}", "--compare", "F,C")
    check(f"a comparison naming a missing arm is refused (exit {rc})", rc != 0 and "both arms" in out)
    dup = arm_dir("Fdup", st3, "F", edit=lambda rows: rows + rows[:1])
    rc, _, out = score("d.json", f"F={dup}", f"C={C}", "--compare", "F,C")
    check(f"a duplicated id is refused (exit {rc})", rc != 0 and "duplicate" in out)
    miss = arm_dir("Fmiss", st3, "F", edit=lambda rows: rows[1:])
    rc, o, out = score("x.json", f"F={miss}", f"C={C}", "--compare", "F,C")
    check(f"a missing id withholds the verdict (exit {rc})", rc != 0 and "missing" in out)

    def rescore(label, published, arms, cmp):
        args = [f"{a}={arm_dir('re-' + label + '-' + a, src)}" for a, src in arms]
        rc, new, _ = score(f"re-{label}.json", *args, "--compare", cmp)
        check(f"re-score {label} identical to the published scores", new == json.load(gzip.open(ROOT / published, "rt")))

    rescore("stage3", "results/llamacpp/stage3/scores.json.gz",
            [("S", st3), ("C", "results/inside4b-full/full-C")], "S,C")
    rescore("stage1", "results/llamacpp/stage1-full/scores.json.gz",
            [("L", "results/llamacpp/stage1-full"), ("R", "results/inside4b-full/full-R")], "L,R")
    rescore("inside4b", "results/inside4b-full/scores.json.gz",
            [("C", "results/inside4b-full/full-C"), ("R", "results/inside4b-full/full-R")], "C,R")
    shutil.rmtree(scr)
    print("FAST SCORER CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
