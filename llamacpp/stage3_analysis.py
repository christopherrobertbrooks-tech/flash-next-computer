"""LLAMACPP-PLAN v3 stage 3: the frozen ANALYSIS identity (Codex stage-2 confirm, finding 6), separate from the
runtime files run_split.py --expect checks. `freeze` writes the sha256 of everything the scoring uses (the scorer, its
answer parser, the saved C comparator, the question file, the launcher and the artifact builder) plus the repo
commit; `verify` must pass before stage 3 is scored, and `score` verifies, then runs the registered comparison.

usage (repo root, Dev-Console):
    python llamacpp/stage3_analysis.py freeze results/llamacpp/stage3-analysis.json
    python llamacpp/stage3_analysis.py verify results/llamacpp/stage3-analysis.json
    python llamacpp/stage3_analysis.py score results/llamacpp/stage3-analysis.json S_RUN_DIR OUT.json
"""
import gzip, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["stage5/score.py", "computer/stage4_requests.py", "inside4b/final-primary.jsonl",
         "results/inside4b-full/full-C/replies.jsonl.gz", "results/inside4b-full/full-C/questions.sha256",
         "llamacpp/stage3.sh", "llamacpp/make_stage3_artifacts.py", "llamacpp/stage3_analysis.py",
         "results/llamacpp/stage3-artifacts.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(path):
    want = json.loads(Path(path).read_text())
    bad = [f for f, h in want["files"].items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    if bad:
        sys.exit(f"analysis identity check failed: {bad}")
    print(f"{len(want['files'])} analysis files match (frozen at {want['commit'][:12]})")
    return want


def main():
    cmd, path = sys.argv[1], Path(sys.argv[2])
    if cmd == "freeze":
        if path.exists():
            sys.exit(f"{path} exists")
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        path.write_text(json.dumps({"commit": commit, "files": {f: sha(ROOT / f) for f in FILES}}, indent=1) + "\n")
        print(f"froze {len(FILES)} analysis files at {commit[:12]}")
    elif cmd == "verify":
        verify(path)
    elif cmd == "score":
        verify(path)
        s_dir, out = Path(sys.argv[3]), sys.argv[4]
        with tempfile.TemporaryDirectory() as tmp:
            c = Path(tmp) / "C"
            c.mkdir()
            shutil.copy(ROOT / "results/inside4b-full/full-C/questions.sha256", c)
            with gzip.open(ROOT / "results/inside4b-full/full-C/replies.jsonl.gz", "rt") as f:
                (c / "replies.jsonl").write_text(f.read())
            r = subprocess.run([sys.executable, str(ROOT / "stage5/score.py"), str(ROOT / "inside4b/final-primary.jsonl"),
                                out, f"S={s_dir}", f"C={c}", "--compare", "S,C"])
        sys.exit(r.returncode)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
