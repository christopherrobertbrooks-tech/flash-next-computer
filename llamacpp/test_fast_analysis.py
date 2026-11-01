"""CPU controls for fast_analysis.py's arm-validity gate (Codex fast build review, finding 5), on real records
(stage 3's 160 S replies relabeled F and G) with synthetic run status files. Each case runs score_all and checks
which comparisons were scored and which were withheld:
- both arms valid: all four comparisons scored, exit 0;
- a lock error on F's last question (all 160 rows present): F withheld (F,C and F,S not scored), G still scored,
  exit nonzero;
- F's helper failed at shutdown after 160 answers (status "incomplete"): F withheld;
- F has no startup.json: F withheld;
- F's startup names a different GGUF: F withheld;
- a text-budget stop is NOT an invalidity (a failed question): F still scored.

usage: python llamacpp/test_fast_analysis.py SCRATCH   (repo root)
"""
import contextlib, gzip, io, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llamacpp"))
import fast_analysis as FA  # noqa: E402


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

    src = [json.loads(l) for l in gzip.open(ROOT / "results/llamacpp/stage3/replies.jsonl.gz", "rt")]

    def arm(name, label, status="ok", edit=None, startup=True, gguf=None):
        d = scr / name
        d.mkdir()
        shutil.copy(ROOT / "results/llamacpp/stage3/questions.sha256", d)
        rows = [dict(r, arm=label) for r in src]
        if edit:
            rows = edit(rows)
        (d / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        man = json.loads((ROOT / f"results/fast/fast-artifacts-{label}.json").read_text())
        if startup:
            (d / "startup.json").write_text(json.dumps({
                "status": status, "reason": None if status == "ok" else "synthetic", "arm": label,
                "lm_input": "tokens", "gguf_sha256": gguf or man["files"][man["gguf"]],
                "artifacts_verified": len(man["files"])}))
        return d

    def lock_last(rows):
        r = dict(rows[-1])
        r["inside"] = dict(r["inside"], lock_error={"at": 9, "got": "V(1)", "want": "V(0)"})
        r["tool_calls"] = list(r["tool_calls"]) + [{"expression": None, "result": None, "error": "x",
                                                    "error_kind": "trace_mismatch"}]
        return rows[:-1] + [r]

    def budget_last(rows):
        r = dict(rows[-1])
        r["run"] = dict(r["run"], stop="text budget")
        return rows[:-1] + [r]

    G = arm("G", "G")

    def run(label, F):
        out = scr / f"out-{label}"
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = FA.score_all(F, G, out)
        scored = sorted(p.stem for p in out.glob("*-vs-*.json"))
        return rc, scored, buf.getvalue()

    rc, sc, _ = run("valid", arm("F-valid", "F"))
    check(f"both valid: exit {rc}, scored {sc}", rc == 0 and sc == ["F-vs-C", "F-vs-S", "G-vs-C", "G-vs-S"])
    for label, F in (("a lock error on F's last question", arm("F-lock", "F", edit=lock_last)),
                     ("F incomplete at shutdown (all rows)", arm("F-inc", "F", status="incomplete")),
                     ("F without startup.json", arm("F-nost", "F", startup=False)),
                     ("F's startup names another GGUF", arm("F-gguf", "F", gguf="0" * 64))):
        rc, sc, out = run(label.split()[0] + str(len(label)), F)
        check(f"{label}: exit {rc}, scored {sc}, F withheld", rc != 0 and sc == ["G-vs-C", "G-vs-S"]
              and "WITHHELD" in out)
    rc, sc, _ = run("budget", arm("F-budget", "F", edit=budget_last))
    check(f"a text-budget stop isn't an invalidity: exit {rc}, scored {sc}", rc == 0 and len(sc) == 4)
    shutil.rmtree(scr)
    print("FAST ANALYSIS CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
