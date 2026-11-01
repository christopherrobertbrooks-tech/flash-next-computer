"""CPU controls for the early-stop build (EARLYSTOP-PLAN v2). Exits 1 if any fails.
A. split_generate's on_decide hook (scripted fakes with a real reference stream, as test_split_chat.py):
   - a recording hook changes nothing (tokens, counts, lock errors equal the run without it, rows and token modes);
   - it's called exactly at the language model's decisions (the prompt's last position and every text step the
     circuit doesn't own), never for chip tokens or forced echoes;
   - refusing at a decision ends the run there: the tokens are the unchanged ones before that index, early_stop is
     set, and the helper isn't called again.
B. run_questions: --early-stop gives stop "early stop" and a decisions.jsonl line with the ids; --record-decisions
   alone changes nothing else; neither passes no hook at all.
C. earlystop_eval on a synthetic run: premature stops, failures stopped, the time estimate, the candidate verdict
   (the final one withheld), the live selection; and it refuses impossible decisions (a negative, NaN, decreasing or
   too-late time, a zero duration, a decided token that isn't the generated one, indices out of order, an index past
   the tokens on an EOS run).
D. earlystop_live_check: a matching valid live run passes and turns the candidate into the final verdict; it fails,
   and the final verdict is withheld, on a different token, a wrong early-stop flag, a time over the tolerance, an
   incomplete live run, a trace error, another runtime manifest, a negative live time.
usage (riscv dir on PYTHONPATH, the qwenputer venv): python test_earlystop.py QWENPUTER_MODEL_DIR SCRATCH
"""
import json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import earlystop_eval as EV  # noqa: E402
import earlystop_live_check as LC  # noqa: E402
import earlystop_rule as RULE  # noqa: E402
import run_split as RS  # noqa: E402
import split_chat as SC  # noqa: E402
import test_split_chat as TS  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402

V0 = TS.V0


def main():
    T = BR.Text(Path(sys.argv[1]) / "tokenizer.json")
    scr = Path(sys.argv[2])
    if scr.exists():
        shutil.rmtree(scr)
    scr.mkdir(parents=True)
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    prompt = T.encode("<|im_start|>user\nWhat is it?<|im_end|>\n<|im_start|>assistant\n")
    one = TS.reference_stream(T, prompt, [("text", "Sure. "), ("call", "84/18"), ("text", "ANSWER: 14/3")])

    def run(mode, gen, hook=None):
        log = []
        c, q = TS.fake_chat(T, TS.World(gen), log)
        TS.core.snapshot = lambda c_, m: None
        lock = BR.Session(T, TS.machine())
        lock.history(prompt)
        r = SC.split_generate(c, TS.FakeHelper(q, log), prompt, TS.machine(), 8192, max_chip=2_000_000, lock=lock,
                              prompt_rows=lambda chat, ids: TS.torch.zeros(len(ids), TS.HID), lm_input=mode,
                              on_decide=hook)
        return r, log

    for mode in ("rows", "tokens"):
        base, _ = run(mode, one)
        seen = []
        rec, _ = run(mode, one, lambda i, t: seen.append((i, t)) and False)
        same = {k: base[k] for k in ("tokens", "text_tokens", "chip_tokens", "lock_error")} == \
               {k: rec[k] for k in ("tokens", "text_tokens", "chip_tokens", "lock_error")}
        gen = base["tokens"]
        w = TS.World(gen)
        idx, qwen_pos = 0, [0]
        for i in range(len(gen) - 1):
            if gen[i] < V0:
                if w.owner[idx] is None:
                    qwen_pos.append(i + 1)
            idx += 1
        check(f"{mode}: a recording hook changes nothing", same and not rec["early_stop"])
        check(f"{mode}: the hook sees exactly the model's decisions ({len(seen)}), each token = the generated one",
              [i for i, _ in seen] == qwen_pos and all(gen[i] == t for i, t in seen if i < len(gen)))
        m = len(seen) // 2
        stop_at = seen[m][0]
        calls = []
        r, log = run(mode, one, lambda i, t: (calls.append(i), len(calls) - 1 == m)[1])
        helper_calls = sum(1 for x in log if x[0] in ("rows", "tokens"))
        fed = 1 + sum(1 for t in gen[:stop_at] if t < V0)   # the prefill, then each text input before the stop
        check(f"{mode}: refusing decision {m} ends the run at index {stop_at}, without that token, and the helper "
              f"isn't called again ({helper_calls} calls)",
              r["early_stop"] and r["tokens"] == gen[:stop_at] and r["lock_error"] is None and helper_calls == fed)

    def fake_gen(tokens, own_idx):
        def g(chat, helper, ids, m, max_new, max_chip=0, lock=None, on_decide=None, **kw):
            out = []
            for i, t in enumerate(tokens):
                if i in own_idx and on_decide is not None and on_decide(i, t):
                    return {"tokens": out, "text_tokens": len(out), "chip_tokens": 0, "lock_error": None,
                            "early_stop": True}
                out.append(t)
            return {"tokens": out, "text_tokens": len(out), "chip_tokens": 0, "lock_error": None}
        return g

    class Chat:
        v0, eos = V0, {248046, 248044}

        def __init__(self):
            self.T = T

        def prompt_ids(self, msgs):
            return T.encode("<|im_start|>user\n" + msgs[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n")

        def text_of(self, toks):
            return T.decode_bytes([t for t in toks if t < V0]).decode(errors="replace")

    span = T.encode(" and so on, the same sentence again and again until it repeats")[:8] * 8
    looping = (span * 7)[:64 * 7] + [248046]
    qs = [{"id": "t-1", "group": "1", "prompt": "loop?"}]
    for label, kw, want_stop, want_file in (("early stop", {"early_stop": True}, "early stop", True),
                                            ("record only", {"record": True}, "eos", True),
                                            ("neither", {}, "eos", False)):
        d = scr / label.replace(" ", "-")
        d.mkdir()
        got_hook = []

        def g(chat, helper, ids, m, max_new, max_chip=0, lock=None, **k):
            got_hook.append("on_decide" in k)
            return fake_gen(looping, set(range(len(looping))))(chat, helper, ids, m, max_new, max_chip, lock, **k)

        st, _ = RS.run_questions(Chat(), None, qs, str(d), 8192, 2_000_000, 16384, g, "F", **kw)
        row = json.loads(open(d / "replies.jsonl").readline())
        f = d / "decisions.jsonl"
        dl = json.loads(f.read_text()) if f.exists() else None
        ok = st == "ok" and row["run"]["stop"] == want_stop and f.exists() == want_file and got_hook == [bool(kw)]
        if dl:
            ok = ok and dl["tokens"] == (looping[:RULE.stop_index(looping)] if kw.get("early_stop") else looping)
        check(f"run_questions, {label}: stop {row['run']['stop']}, decisions file {f.exists()}, hook passed {got_hook}",
              ok)

    # C: the evaluator on a synthetic run (4 questions), with its validation
    import copy, hashlib
    manifest = scr / "manifest.json"
    manifest.write_text(json.dumps({"files": {"a": "0" * 64, "b": "1" * 64}}))
    msha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rep = list(range(1000, 1064)) * 6 + [5]                # stops at its 6th repeat (own index 383)
    plain = list(range(2000, 2100)) + [248046]
    cases = [("q1", rep, True, "eos", 40.0), ("q2", rep, False, "text budget", 90.0),
             ("q3", plain, True, "eos", 10.0), ("q4", plain, False, "eos", 12.0)]

    def write_run(d, rows, early_stop, qfile=None, status="ok", extra_row=None, msha_=None):
        d.mkdir()
        with open(d / "replies.jsonl", "w") as fr, open(d / "decisions.jsonl", "w") as fd:
            for qid, toks, right, stop, secs, decs, es in rows:
                row = {"id": qid, "seconds": secs, "run": {"stop": stop}, "inside": {"text_tokens": len(toks)},
                       "tool_calls": []}
                if extra_row and qid in extra_row:
                    row = extra_row[qid](row)
                fr.write(json.dumps(row) + "\n")
                fd.write(json.dumps({"id": qid, "early_stop": es, "tokens": toks, "decisions": decs,
                                     "seconds_monotonic": secs}) + "\n")
        (d / "startup.json").write_text(json.dumps({"status": status, "reason": None, "arm": "F", "lm_input": "tokens",
                                                    "record_decisions": True, "early_stop": early_stop, "max_new": 8192,
                                                    "max_chip": 2_000_000, "n_ctx": 16384, "artifacts_verified": 2,
                                                    "expect_sha256": msha_ or msha}))
        if qfile:
            (d / "questions.sha256").write_text(hashlib.sha256(qfile.read_bytes()).hexdigest() + "  q\n")

    def decs(toks, secs):
        return [[i, t, round(secs * (i + 1) / (len(toks) + 1), 6)] for i, t in enumerate(toks)]

    base_rows = [(c[0], c[1], c[2], c[3], c[4], decs(c[1], c[4]), False) for c in cases]
    U = scr / "unchanged"
    write_run(U, base_rows, False)
    (U / "scores.json").write_text(json.dumps({"rows": [{"id": c[0], "correct": c[2]} for c in cases]}))
    s, per, live = EV.evaluate(U, U / "scores.json")
    est = {p["id"]: p["seconds_with_rule"] for p in per}
    check(f"eval: premature {s['premature_ids']}, failures stopped {s['failures_stopped']}, budget failures stopped "
          f"{s['budget_failures_stopped']}, candidate {s['candidate_verdict']}, final {s['final_verdict']}",
          s["premature_ids"] == ["q1"] and s["failures_stopped"] == 1 and s["budget_failures_stopped"] == 1
          and s["candidate_verdict"] == "candidate don't keep" and s["final_verdict"].startswith("withheld")
          and abs(est["q1"] - 40.0 * 384 / 386) < 1e-3 and est["q3"] == 10.0)
    check(f"eval: live selection {live}", live["rule_stops"] == ["q1", "q2"] and live["text_budget"] == []
          and live["shortfall"]["text_budget"] == 2 and set(live["eos_median_and_longest"]) <= {"q3", "q4"})

    def bad_eval(label, edit):
        d = scr / ("eval-" + label.replace(" ", "-"))
        rows = copy.deepcopy(base_rows)
        rows[0] = edit(list(rows[0]))
        write_run(d, rows, False)
        shutil.copy(U / "scores.json", d)
        try:
            EV.evaluate(d, d / "scores.json")
            got = "accepted"
        except ValueError as e:
            got = str(e)
        check(f"eval refuses {label} ({got[:60]})", got != "accepted")

    def set_time(n, t):
        def f(r):
            r[5] = copy.deepcopy(r[5])
            r[5][n][2] = t
            return tuple(r)
        return f

    bad_eval("a negative time", set_time(3, -100.0))
    bad_eval("a NaN time", set_time(3, float("nan")))
    bad_eval("a decreasing time", set_time(5, 0.0001))
    bad_eval("a time past the duration", set_time(10, 1e6))
    bad_eval("a zero duration", lambda r: tuple(r[:4] + [0.0] + r[5:]))
    bad_eval("a decided token that isn't the generated one",
             lambda r: tuple(r[:5] + [[[i, (t + 1 if i == 7 else t), x] for i, t, x in r[5]]] + r[6:]))
    bad_eval("a repeated index (times still in order)",
             lambda r: tuple(r[:5] + [[r[5][0], [r[5][0][0], r[5][1][1], r[5][1][2]]] + r[5][2:]] + r[6:]))
    bad_eval("an index past the tokens on an EOS run",
             lambda r: tuple(r[:5] + [r[5] + [[len(r[1]) + 3, 9, r[4] * 0.999]]] + r[6:]))

    # D: the live check and the final verdict
    out = scr / "score-out"
    out.mkdir()
    for case, cand in (("dont", s), ("keep", dict(s, candidate_verdict="candidate keep"))):
        (out / f"eval-{case}.json").write_text(json.dumps({"summary": cand, "live_check_ids": live,
                                                           "per_question": per}))
    sel = [i for g in ("rule_stops", "text_budget", "eos_median_and_longest") for i in live[g]]
    lq = out / "live-questions.jsonl"
    lq.write_text("".join(json.dumps({"id": i}) + "\n" for i in sel))

    def live_run(name, mutate=None, slow=0.0, **kw):
        rows = []
        for p in per:
            if p["id"] not in sel:
                continue
            toks = next(c[1] for c in cases if c[0] == p["id"])
            toks = toks[:p["stop_gen_index"]] if p["rule_stops"] else toks
            secs = p["seconds_with_rule"] + slow
            es = p["rule_stops"]
            stop = "early stop" if es else p["stop"]
            d = [[i, t, round(secs * (i + 1) / (len(toks) + 2), 6)] for i, t in enumerate(toks)]
            if es:
                d.append([len(toks), 1000 + 0, round(secs * 0.999, 6)])   # the refused, unappended decision
            row = [p["id"], toks, p["right"], stop, secs, d, es]
            if mutate:
                row = mutate(row)
            rows.append(tuple(row))
        L = scr / name
        write_run(L, rows, True, qfile=lq, **kw)
        return L

    def final_of(L, case="dont"):
        return LC.check(U, out / f"eval-{case}.json", lq, L, manifest)

    f = final_of(live_run("live-ok", slow=0.5))
    check(f"live check: a matching valid run passes; final verdict {f['final_verdict']}",
          f["live_check_passed"] and f["final_verdict"] == "don't keep")
    f = final_of(live_run("live-ok-keep", slow=0.5), "keep")
    check(f"live check passes with a candidate keep: final verdict {f['final_verdict']}", f["final_verdict"] == "keep")
    for label, L in (
            ("one different token", live_run("live-tok", lambda r: r[:1] + [r[1][:-1] + [7]] + r[2:] if r[0] == "q3" else r)),
            ("a wrong early-stop flag", live_run("live-flag", lambda r: r[:6] + [not r[6]] if r[0] == "q4" else r)),
            ("a time over the tolerance", live_run("live-slow", slow=30.0)),
            ("an incomplete live run (shutdown failed)", live_run("live-inc", status="incomplete")),
            ("a trace error in the live rows", live_run("live-trace", extra_row={"q3": lambda row: dict(
                row, tool_calls=[{"error_kind": "trace_mismatch"}])})),
            ("a live run against another runtime manifest", live_run("live-man", msha_="f" * 64)),
            ("a negative live time", live_run("live-neg", lambda r: r[:5] + [[[i, t, -1.0] for i, t, _ in r[5]]] + r[6:]
                                                if r[0] == "q3" else r))):
        f = final_of(L, "keep")
        check(f"live check fails on {label}; final verdict {f['final_verdict']}",
              not f["live_check_passed"] and f["final_verdict"].startswith("withheld"))
    shutil.rmtree(scr)
    print("EARLY-STOP CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
