"""CPU controls for the early-stop build (EARLYSTOP-PLAN v2). Exits 1 if any fails.
A. split_generate's on_decide hook (scripted fakes with a real reference stream, as test_split_chat.py):
   - a recording hook changes nothing (tokens, counts, lock errors equal the run without it, rows and token modes);
   - it's called exactly at the language model's decisions (the prompt's last position and every text step the
     circuit doesn't own), never for chip tokens or forced echoes;
   - refusing at a decision ends the run there: the tokens are the unchanged ones before that index, early_stop is
     set, and the helper isn't called again.
B. run_questions: --early-stop gives stop "early stop" and a decisions.jsonl line with the ids; --record-decisions
   alone changes nothing else; neither passes no hook at all.
C. earlystop_eval on a synthetic run: premature stops, failures stopped, the time estimate, the verdict, the live
   selection.
D. earlystop_live_check: a matching live run passes; a different token, a wrong early-stop flag, or a time over the
   tolerance fails.
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

    # C: the evaluator on a synthetic run (4 questions)
    d = scr / "eval"
    d.mkdir()
    rep = list(range(1000, 1064)) * 6 + [5]                # stops at its 6th repeat (own index 383)
    plain = list(range(2000, 2100)) + [248046]
    cases = [("q1", rep, True, "eos", 40.0), ("q2", rep, False, "text budget", 90.0),
             ("q3", plain, True, "eos", 10.0), ("q4", plain, False, "eos", 12.0)]
    with open(d / "replies.jsonl", "w") as fr, open(d / "decisions.jsonl", "w") as fd:
        for qid, toks, right, stop, secs in cases:
            fr.write(json.dumps({"id": qid, "seconds": secs, "run": {"stop": stop},
                                 "inside": {"text_tokens": len(toks)}}) + "\n")
            fd.write(json.dumps({"id": qid, "early_stop": False, "tokens": toks,
                                 "decisions": [[i, t, round(secs * (i + 1) / len(toks), 6)] for i, t in enumerate(toks)]})
                     + "\n")
    (d / "scores.json").write_text(json.dumps({"rows": [{"id": c[0], "correct": c[2]} for c in cases]}))
    s, per, live = EV.evaluate(d, d / "scores.json")
    est = {p["id"]: p["seconds_with_rule"] for p in per}
    check(f"eval: premature {s['premature_ids']}, failures stopped {s['failures_stopped']}, budget failures stopped "
          f"{s['budget_failures_stopped']}, verdict {s['verdict']}",
          s["premature_ids"] == ["q1"] and s["failures_stopped"] == 1 and s["budget_failures_stopped"] == 1
          and s["verdict"] == "don't keep" and abs(est["q1"] - 40.0 * 384 / 385) < 1e-3 and est["q3"] == 10.0)
    check(f"eval: live selection {live}", live["rule_stops"] == ["q1", "q2"] and live["text_budget"] == []
          and live["shortfall"]["text_budget"] == 2 and set(live["eos_median_and_longest"]) <= {"q3", "q4"})

    # D: the live check
    ev_path = scr / "eval.json"
    ev_path.write_text(json.dumps({"summary": s, "live_check_ids": live, "per_question": per}))

    def live_dir(name, mutate=None, slow=0.0):
        L = scr / name
        L.mkdir()
        with open(L / "replies.jsonl", "w") as fr, open(L / "decisions.jsonl", "w") as fd:
            for p in per:
                toks = next(c[1] for c in cases if c[0] == p["id"])
                toks = toks[:p["stop_gen_index"]] if p["rule_stops"] else toks
                es = p["rule_stops"]
                if mutate:
                    toks, es = mutate(p["id"], list(toks), es)
                fr.write(json.dumps({"id": p["id"], "seconds": p["seconds_with_rule"] + slow}) + "\n")
                fd.write(json.dumps({"id": p["id"], "early_stop": es, "tokens": toks}) + "\n")
        return L

    passes = lambda L: all(ok for _, ok, _ in LC.check(d, ev_path, L)[0])
    check("live check: a matching live run passes", passes(live_dir("live-ok", slow=0.5)))
    check("live check: one different token fails",
          not passes(live_dir("live-tok", lambda q, t, e: (t[:-1] + [7] if q == "q1" else t, e))))
    check("live check: a wrong early-stop flag fails",
          not passes(live_dir("live-flag", lambda q, t, e: (t, (not e) if q == "q2" else e))))
    check("live check: a time over the tolerance fails", not passes(live_dir("live-slow", slow=30.0)))
    shutil.rmtree(scr)
    print("EARLY-STOP CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
