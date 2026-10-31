"""CPU regression checks for stage5/score.py's arm handling (Codex stage-2 review finding 4), on real records:
the 12 stage-2 pilot rows (arm S) and inside-4b C's rows for the same 12 questions, with a subset question file.
- `--compare S,C` scores both arms (12 rows each), gives the paired table, the signed S_minus_C interval and the
  S-relative latency rule;
- an unknown arm label, a --compare naming an arm that wasn't given, and an unsupported first arm all fail
  (nonzero exit) instead of scoring a subset silently;
- inside-4b (C vs R) and stage 1 (L vs R) re-score identically to their published scores.

usage: python llamacpp/test_score_arms.py SCRATCH   (from the repo root)
"""
import gzip, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE = ROOT / "stage5" / "score.py"


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

    def score(*args):
        r = subprocess.run([sys.executable, str(SCORE), *map(str, args)], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    S = [json.loads(l) for l in open(ROOT / "results/llamacpp/stage2-pilot/replies.jsonl")]
    ids = {r["id"] for r in S}
    qs = [json.loads(l) for l in open(ROOT / "inside4b/final-primary.jsonl") if json.loads(l)["id"] in ids]
    qf = scr / "q12.jsonl"
    qf.write_text("".join(json.dumps(q) + "\n" for q in qs))
    h = hashlib.sha256(qf.read_bytes()).hexdigest()
    for arm, rows in (("S", S), ("C", [json.loads(l) for l in gzip.open(ROOT / "results/inside4b-full/full-C/replies.jsonl.gz", "rt")
                                        if json.loads(l)["id"] in ids])):
        d = scr / arm
        d.mkdir()
        (d / "questions.sha256").write_text(f"{h}  q12.jsonl\n")
        (d / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rc, out = score(qf, scr / "o.json", f"S={scr / 'S'}", f"C={scr / 'C'}", "--compare", "S,C")
    o = json.load(open(scr / "o.json"))["summary"] if rc == 0 else {}
    p = o.get("primary", {})
    pairs = p.get("pairs", {})
    check(f"--compare S,C: exit {rc}, both arms scored, S vs C, S_minus_C {p.get('S_minus_C')}, pairs {pairs}",
          rc == 0 and set(o.get("arms", {})) == {"S", "C"} and p.get("compared") == "S vs C"
          and "S_minus_C" in p and sum(pairs.values()) == 12 and "ci95" in p
          and set(p.get("median_seconds", {})) == {"S", "C"})
    rc, out = score(qf, scr / "o2.json", f"S={scr / 'S'}", f"X={scr / 'C'}")
    check(f"an unknown arm label fails (exit {rc})", rc != 0 and "unknown arm" in out)
    rc, out = score(qf, scr / "o3.json", f"C={scr / 'C'}", "--compare", "S,C")
    check(f"--compare naming a missing arm fails (exit {rc})", rc != 0 and "both arms" in out)
    rc, out = score(qf, scr / "o4.json", f"S={scr / 'S'}", f"C={scr / 'C'}", "--compare", "R,C")
    check(f"an unsupported first arm fails (exit {rc})", rc != 0 and "first arm" in out)

    def rescore(label, published, arms, cmp):
        dirs = []
        for a, src in arms:
            d = scr / f"re-{label}-{a}"
            d.mkdir()
            shutil.copy(src / "questions.sha256", d)
            with gzip.open(src / "replies.jsonl.gz", "rt") as f:
                (d / "replies.jsonl").write_text(f.read())
            dirs.append(f"{a}={d}")
        rc, _ = score(ROOT / "inside4b/final-primary.jsonl", scr / f"re-{label}.json", *dirs, "--compare", cmp)
        new = json.load(open(scr / f"re-{label}.json")) if rc == 0 else None
        old = json.load(gzip.open(published, "rt"))
        check(f"re-score {label} identical to the published scores (summary and rows)", new == old)

    rescore("inside4b", ROOT / "results/inside4b-full/scores.json.gz",
            [("C", ROOT / "results/inside4b-full/full-C"), ("R", ROOT / "results/inside4b-full/full-R")], "C,R")
    rescore("stage1", ROOT / "results/llamacpp/stage1-full/scores.json.gz",
            [("L", ROOT / "results/llamacpp/stage1-full"), ("R", ROOT / "results/inside4b-full/full-R")], "L,R")
    shutil.rmtree(scr)
    print("SCORER ARM CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
