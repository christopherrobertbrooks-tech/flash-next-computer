"""CPU controls for run_split.py's run validity and memory sampling (Codex stage-2 review findings 2 and 5).
- run_questions: a normal run is "ok"; a lock error (even with a right-looking final text) stops the run as
  "invalid design" after saving that row, with no later rows; an exception stops it as "incomplete" with a
  harness_error row; a prompt too long for the helper's context is refused before generating;
- parse_apps keeps only the V100's entries (a pid on two GPUs, another GPU's processes);
- MemSampler: a missing pid during "run" is unavailable (never 0); the helper's absence during "shutdown" is
  expected; finish() joins and takes a final sample; the report's max is the max of per-sample sums;
- verify_artifacts refuses a changed file;
- the early-stop setting: 'default' is on only for arm F in token mode, explicit on/off win, and the registered
  launchers pass it explicitly.

usage (riscv dir on PYTHONPATH, the qwenputer venv): python test_run_split.py QWENPUTER_MODEL_DIR SCRATCH
"""
import json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_split as RS  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402

U = RS.V100_UUID


def main():
    model, scratch = Path(sys.argv[1]), Path(sys.argv[2])
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    T = BR.Text(model / "tokenizer.json")
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    class Chat:
        v0, eos = 248320, {248046, 248044}

        def __init__(self, n_prompt=20):
            self.T, self.n = T, n_prompt

        def prompt_ids(self, msgs):
            return T.encode("<|im_start|>user\n" + msgs[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n")[:self.n] \
                if self.n < 1000 else [1] * self.n

        def text_of(self, toks):
            return T.decode_bytes([t for t in toks if t < self.v0]).decode(errors="replace")

    answer = T.encode("ANSWER: 4") + [248046]

    def gen_ok(chat, h, ids, m, max_new, max_chip=0, lock=None):
        return {"tokens": answer, "text_tokens": len(answer) - 1, "chip_tokens": 0, "lock_error": None}

    def gen_lock(chat, h, ids, m, max_new, max_chip=0, lock=None):
        return {"tokens": answer, "text_tokens": 3, "chip_tokens": 5,
                "lock_error": {"at": 4, "got": "V(1)", "want": "V(0)"}}

    def gen_raise(chat, h, ids, m, max_new, max_chip=0, lock=None):
        raise RuntimeError("helper: non-finite logit at id 7")

    qs = [{"id": f"t-{i}", "group": "1", "prompt": f"What is {i}+{i}?"} for i in range(3)]

    def run(label, gen, chat=None, max_new=8192):
        d = scratch / label
        d.mkdir()
        st, why = RS.run_questions(chat or Chat(), None, qs, str(d), max_new, 2_000_000, 16384, gen)
        rows = [json.loads(l) for l in open(d / "replies.jsonl")] if (d / "replies.jsonl").exists() else []
        return st, why, rows

    st, why, rows = run("ok", gen_ok)
    check(f"normal: {st}, {len(rows)} rows", st == "ok" and len(rows) == 3)
    st, why, rows = run("lock", gen_lock)
    check(f"a lock error (final text 'ANSWER: 4'): {st}, {len(rows)} row(s), stop {rows[0]['run']['stop'] if rows else None}",
          st == "invalid design" and len(rows) == 1 and rows[0]["inside"]["lock_error"]
          and rows[0]["tool_calls"][-1]["error_kind"] == "trace_mismatch")
    st, why, rows = run("raise", gen_raise)
    check(f"an exception: {st}, {len(rows)} row(s), harness_error {rows[0]['harness_error'] if rows else None}",
          st == "incomplete" and len(rows) == 1 and "non-finite" in rows[0]["harness_error"])
    st, why, rows = run("ctx", gen_ok, chat=Chat(n_prompt=9000))
    check(f"a prompt that can't fit with the budget: {st} ({why[:60]}), {len(rows)} rows",
          st == "incomplete" and len(rows) == 0 and "context" in why)

    text = "\n".join([f"{U}, 100, 1500", f"{U}, 200, 15000", "GPU-other, 100, 1400", "GPU-other, 300, 7000",
                      f"{U}, 100, 200"])
    got = RS.parse_apps(text, U)
    check(f"parse_apps: only the V100's entries, a pid's V100 entries summed ({got})",
          got == {"100": 1700, "200": 15000})

    feed = []
    sampler = RS.MemSampler(U, {"parent": 100, "helper": 200}, every=0.01, query=lambda: feed[0])
    feed.append(f"{U}, 100, 1500\n{U}, 200, 15000")
    sampler.sample()
    feed[0] = f"{U}, 100, 1600"
    sampler.sample()                                       # run phase: the helper missing -> unavailable
    sampler.phase = "shutdown"
    sampler.sample()                                       # shutdown: the helper's exit is expected
    sampler.start()
    sampler.finish()
    rep = sampler.report()
    check(f"MemSampler: run-phase gap unavailable, shutdown ok; max of sums {rep['sampled_max_sum_mib']}, "
          f"unavailable {rep['samples_unavailable']}, joined {not sampler.is_alive()}",
          sampler.samples[1].get("unavailable") and "sum" not in sampler.samples[1]
          and sampler.samples[2].get("sum") == 1600 and rep["sampled_max_sum_mib"] == 16500
          and rep["samples_unavailable"] == 1 and not sampler.is_alive())
    f = scratch / "a.bin"
    f.write_bytes(b"x")
    e = scratch / "expect.json"
    e.write_text(json.dumps({"files": {str(f): "0" * 64}}))
    try:
        RS.verify_artifacts(e)
        got = "accepted"
    except SystemExit as x:
        got = str(x)
    check(f"verify_artifacts refuses a changed file ({got[:50]})", "identity check failed" in got)
    e.write_text(json.dumps({"files": {str(f): RS.RQ.sha(f)}}))
    check("verify_artifacts accepts the right hash", RS.verify_artifacts(e) == 1)
    # the early-stop default (adopted for arm F after results/earlystop-results.md)
    cases = [(("default", "F", "tokens"), True), (("default", "G", "tokens"), False), (("default", "S", "rows"), False),
             (("default", "F", "rows"), False), (("off", "F", "tokens"), False), (("on", "G", "tokens"), True),
             (("on", "S", "rows"), True)]
    got = [(c, RS.resolve_early_stop(*c)) for c, _ in cases]
    check(f"early-stop setting: default on only for F in token mode; explicit on/off win ({got})",
          all(RS.resolve_early_stop(*c) == w for c, w in cases))
    here = Path(RS.__file__).parent
    for name in ("fast.sh", "earlystop.sh"):
        lines = [l for l in (here / name).read_text().splitlines() if "run_split.py" in l or "--lm-input" in l]
        runs = [l for l in lines if "--lm-input" in l]
        check(f"{name}: every run passes --early-stop explicitly ({len(runs)} run lines)",
              runs and all("--early-stop on" in l or "--early-stop off" in l for l in runs))
    shutil.rmtree(scratch)
    print("RUN_SPLIT CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
