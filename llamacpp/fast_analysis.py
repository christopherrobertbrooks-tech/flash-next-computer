"""FAST-PLAN: the frozen analysis identity for the fast runs (arms F and G), separate from stage 3's (which stays
unchanged). `freeze` writes the sha256 of everything the scoring uses (the scorer and its parser, this wrapper, C's
and S's saved replies and question hashes, the questions, the launcher, the artifact builder and both runtime
manifests) plus the repo commit. `verify` must pass before scoring. `score` verifies, then runs every registered
comparison: F,C (primary), G,C, F,S, G,S (secondary), each with stage5/score.py --compare.

usage (repo root, Dev-Console):
    python llamacpp/fast_analysis.py freeze results/fast/fast-analysis.json
    python llamacpp/fast_analysis.py verify results/fast/fast-analysis.json
    python llamacpp/fast_analysis.py score results/fast/fast-analysis.json F_RUN_DIR G_RUN_DIR OUT_DIR
"""
import gzip, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["stage5/score.py", "computer/stage4_requests.py", "llamacpp/fast_analysis.py", "inside4b/final-primary.jsonl",
         "results/inside4b-full/full-C/replies.jsonl.gz", "results/inside4b-full/full-C/questions.sha256",
         "results/llamacpp/stage3/replies.jsonl.gz", "results/llamacpp/stage3/questions.sha256",
         "llamacpp/fast.sh", "llamacpp/make_fast_artifacts.py",
         "results/fast/fast-artifacts-F.json", "results/fast/fast-artifacts-G.json"]
COMPARISONS = [("F", "C", "primary"), ("G", "C", "secondary"), ("F", "S", "secondary"), ("G", "S", "secondary")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(path):
    want = json.loads(Path(path).read_text())
    bad = [f for f, h in want["files"].items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    if bad:
        sys.exit(f"analysis identity check failed: {bad}")
    print(f"{len(want['files'])} analysis files match (frozen at {want['commit'][:12]})")


def saved(tmp, arm, src):
    d = Path(tmp) / arm
    d.mkdir()
    shutil.copy(ROOT / src / "questions.sha256", d)
    with gzip.open(ROOT / src / "replies.jsonl.gz", "rt") as f:
        (d / "replies.jsonl").write_text(f.read())
    return d


def main():
    cmd, path = sys.argv[1], Path(sys.argv[2])
    if cmd == "freeze":
        if path.exists():
            sys.exit(f"{path} exists")
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        path.write_text(json.dumps({"commit": commit, "files": {f: sha(ROOT / f) for f in FILES},
                                    "comparisons": COMPARISONS}, indent=1) + "\n")
        print(f"froze {len(FILES)} analysis files at {commit[:12]}")
    elif cmd == "verify":
        verify(path)
    elif cmd == "score":
        verify(path)
        f_dir, g_dir, out = Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
        out.mkdir(parents=True, exist_ok=False)
        rc_all = 0
        with tempfile.TemporaryDirectory() as tmp:
            dirs = {"F": f_dir, "G": g_dir, "C": saved(tmp, "C", "results/inside4b-full/full-C"),
                    "S": saved(tmp, "S", "results/llamacpp/stage3")}
            for a, b, role in COMPARISONS:
                if not (dirs[a] / "replies.jsonl").exists():
                    print(f"{a},{b} ({role}): no {a} run to score (reported as not run)")
                    rc_all = rc_all or 3
                    continue
                r = subprocess.run([sys.executable, str(ROOT / "stage5/score.py"),
                                    str(ROOT / "inside4b/final-primary.jsonl"), str(out / f"{a}-vs-{b}.json"),
                                    f"{a}={dirs[a]}", f"{b}={dirs[b]}", "--compare", f"{a},{b}"],
                                   capture_output=True, text=True)
                (out / f"{a}-vs-{b}.out").write_text(r.stdout + r.stderr)
                print(f"{a},{b} ({role}): exit {r.returncode}")
                rc_all = rc_all or r.returncode
        sys.exit(rc_all)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
