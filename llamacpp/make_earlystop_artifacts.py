"""EARLYSTOP-PLAN v2: the frozen runtime identity for the early-stop runs (arm F's runtime as in the fast run, plus the
frozen rule, the launcher and the fresh questions), checked by run_split.py --expect. Refuses to overwrite.

usage (on the gateway): python make_earlystop_artifacts.py OUT.json
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_fast_artifacts as MF  # noqa: E402


def main():
    out = Path(sys.argv[1])
    if out.exists():
        sys.exit(f"{out} exists")
    tmp = out.with_suffix(".base.json")
    sys.argv = ["", "F", str(tmp)]
    MF.main()
    rec = json.loads(tmp.read_text())
    tmp.unlink()
    W = MF.W
    for p in (W / "earlystop_rule.py", W / "earlystop.sh", W / "fresh-questions.jsonl"):
        rec["files"][str(p)] = MF.sha(p)
    rec["files"].pop(str(W / "fast.sh"), None)
    rec["experiment"] = "earlystop"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"earlystop: {len(rec['files'])} files")


if __name__ == "__main__":
    main()
