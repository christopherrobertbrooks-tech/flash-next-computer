"""EARLYSTOP-PLAN v2: the frozen analysis identity for the early-stop experiment. `freeze` writes the sha256 of the
scorer and its parser, this tool, the evaluator, the live check, the frozen rule, the fresh questions and the
generator, the launcher and the runtime manifest, plus the repo commit; `verify` checks them.
`score RUN_DIR OUT_DIR`: verify, score the unchanged run (stage5/score.py, arm F alone, on the fresh questions),
check the run's validity (status ok, arm F, token mode, decisions recorded, no lock/trace/harness error), apply the
rule offline (earlystop_eval.py) and write the live-check question file. `live UNCHANGED_RUN EVAL.json LIVE_RUN`:
verify, then the live check.

usage (repo root): python llamacpp/earlystop_analysis.py freeze|verify results/earlystop/earlystop-analysis.json
                   python llamacpp/earlystop_analysis.py score results/earlystop/earlystop-analysis.json RUN_DIR OUT_DIR
                   python llamacpp/earlystop_analysis.py live results/earlystop/earlystop-analysis.json RUN EVAL LIVE_RUN
"""
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llamacpp"))
FILES = ["stage5/score.py", "computer/stage4_requests.py", "stage5/generate.py", "llamacpp/earlystop_analysis.py",
         "llamacpp/earlystop_eval.py", "llamacpp/earlystop_live_check.py", "llamacpp/earlystop_rule.py",
         "llamacpp/earlystop.sh", "llamacpp/make_earlystop_artifacts.py", "results/earlystop/fresh-questions.jsonl",
         "results/earlystop/earlystop-artifacts.json"]
Q = ROOT / "results/earlystop/fresh-questions.jsonl"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(path):
    want = json.loads(Path(path).read_text())
    bad = [f for f, h in want["files"].items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    if bad:
        sys.exit(f"analysis identity check failed: {bad}")
    print(f"{len(want['files'])} analysis files match (frozen at {want['commit'][:12]})")


def validity(run_dir):
    info = json.loads((run_dir / "startup.json").read_text())
    probs = []
    if info.get("status") != "ok":
        probs.append(f"status {info.get('status')} ({info.get('reason')})")
    if info.get("arm") != "F" or info.get("lm_input") != "tokens" or not info.get("record_decisions"):
        probs.append("not arm F in token mode with recorded decisions")
    man = json.loads((ROOT / "results/earlystop/earlystop-artifacts.json").read_text())
    if info.get("artifacts_verified") != len(man["files"]):
        probs.append("artifacts not all verified")
    for l in open(run_dir / "replies.jsonl"):
        r = json.loads(l)
        if (r.get("inside") or {}).get("lock_error") or r.get("harness_error") or \
                any(c.get("error_kind") == "trace_mismatch" for c in r.get("tool_calls") or []):
            probs.append(f"{r['id']}: lock, trace or harness error")
    return probs


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
        run_dir, out = Path(sys.argv[3]), Path(sys.argv[4])
        out.mkdir(parents=True, exist_ok=False)
        probs = validity(run_dir)
        (out / "validity.json").write_text(json.dumps({"valid": not probs, "problems": probs}, indent=1) + "\n")
        if probs:
            sys.exit(f"WITHHELD: {probs[:3]}")
        r = subprocess.run([sys.executable, str(ROOT / "stage5/score.py"), str(Q), str(out / "scores.json"),
                            f"F={run_dir}"], capture_output=True, text=True)
        (out / "scores.out").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            sys.exit(f"scoring failed (exit {r.returncode})")
        import earlystop_eval as EV
        summary, per, live = EV.evaluate(run_dir, out / "scores.json")
        (out / "eval.json").write_text(json.dumps({"summary": summary, "live_check_ids": live, "per_question": per},
                                                  indent=1) + "\n")
        ids = [i for g in ("rule_stops", "text_budget", "eos_median_and_longest") for i in live[g]]
        qs = [json.loads(l) for l in open(Q)]
        (out / "live-questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs if q["id"] in ids))
        print(json.dumps(summary, indent=1))
        print("live-check questions:", ids)
    elif cmd == "live":
        verify(path)
        import earlystop_live_check as LC
        results, shortfall = LC.check(*sys.argv[3:6])
        for qid, ok, why in results:
            print(f"{'ok ' if ok else 'BAD'} {qid}: {why}")
        print("shortfall by category:", shortfall)
        good = results and all(ok for _, ok, _ in results)
        print("LIVE CHECK", "PASSED" if good else "FAILED")
        sys.exit(0 if good else 1)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
