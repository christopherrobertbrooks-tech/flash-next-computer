"""FAST-PLAN: the frozen analysis identity for the fast runs (arms F and G), separate from stage 3's (which stays
unchanged). `freeze` writes the sha256 of everything the scoring uses (the scorer and its parser, this wrapper, C's
and S's saved replies and question hashes, the questions, the launcher, the artifact builder and both runtime
manifests) plus the repo commit. `verify` must pass before scoring. `score` verifies, then runs every registered
comparison: F,C (primary), G,C, F,S, G,S (secondary), each with stage5/score.py --compare.

Arm validity (Codex fast build review, finding 5): before an arm's comparisons run, its run must be valid: its
startup.json says status "ok", arm F/G and lm_input "tokens", with the GGUF sha256 its frozen runtime manifest names
and every manifest file verified; and no row has a lock error, a trace-mismatch record or a harness error. An
invalid or incomplete arm is reported as WITHHELD (its data kept), its comparisons aren't scored, the other arm's
are still scored (never promoted), and the exit is nonzero. Ordinary budget stops are failed questions, not invalid.

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


def arm_problems(arm, run_dir):
    """Why a new arm's run can't be scored (empty: valid)."""
    run_dir = Path(run_dir)
    st, rows = run_dir / "startup.json", run_dir / "replies.jsonl"
    if not st.exists():
        return ["no startup.json"]
    if not rows.exists():
        return ["no replies.jsonl"]
    info = json.loads(st.read_text())
    man = json.loads((ROOT / f"results/fast/fast-artifacts-{arm}.json").read_text())
    probs = []
    if info.get("status") != "ok":
        probs.append(f"run status {info.get('status')!r} ({info.get('reason')})")
    if info.get("arm") != arm or info.get("lm_input") != "tokens":
        probs.append(f"arm {info.get('arm')!r} / lm_input {info.get('lm_input')!r}")
    if info.get("gguf_sha256") != man["files"].get(man["gguf"]):
        probs.append("GGUF sha256 differs from the frozen runtime manifest's")
    if info.get("artifacts_verified") != len(man["files"]):
        probs.append(f"artifacts verified {info.get('artifacts_verified')} of {len(man['files'])}")
    for l in open(rows):
        r = json.loads(l)
        if (r.get("inside") or {}).get("lock_error") or r.get("harness_error") or \
                any(c.get("error_kind") == "trace_mismatch" for c in r.get("tool_calls") or []):
            probs.append(f"{r['id']}: lock error, trace mismatch or harness error")
    return probs


def saved(tmp, arm, src):
    d = Path(tmp) / arm
    d.mkdir()
    shutil.copy(ROOT / src / "questions.sha256", d)
    with gzip.open(ROOT / src / "replies.jsonl.gz", "rt") as f:
        (d / "replies.jsonl").write_text(f.read())
    return d


def score_all(f_dir, g_dir, out):
    """Validate both arms, then run every registered comparison of the valid ones. Returns the exit status."""
    f_dir, g_dir, out = Path(f_dir), Path(g_dir), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rc_all = 0
    bad = {a: arm_problems(a, d) for a, d in (("F", f_dir), ("G", g_dir))}
    for a, p in bad.items():
        (out / f"{a}-validity.json").write_text(json.dumps({"arm": a, "valid": not p, "problems": p}, indent=1) + "\n")
        print(f"arm {a}: {'valid' if not p else 'WITHHELD: ' + '; '.join(p[:3])}")
    with tempfile.TemporaryDirectory() as tmp:
        dirs = {"F": f_dir, "G": g_dir, "C": saved(tmp, "C", "results/inside4b-full/full-C"),
                "S": saved(tmp, "S", "results/llamacpp/stage3")}
        for a, b, role in COMPARISONS:
            if bad[a]:
                print(f"{a},{b} ({role}): WITHHELD (arm {a} isn't valid); not scored")
                rc_all = rc_all or 4
                continue
            r = subprocess.run([sys.executable, str(ROOT / "stage5/score.py"),
                                str(ROOT / "inside4b/final-primary.jsonl"), str(out / f"{a}-vs-{b}.json"),
                                f"{a}={dirs[a]}", f"{b}={dirs[b]}", "--compare", f"{a},{b}"],
                               capture_output=True, text=True)
            (out / f"{a}-vs-{b}.out").write_text(r.stdout + r.stderr)
            print(f"{a},{b} ({role}): exit {r.returncode}")
            rc_all = rc_all or r.returncode
    return rc_all


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
        sys.exit(score_all(sys.argv[3], sys.argv[4], sys.argv[5]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
