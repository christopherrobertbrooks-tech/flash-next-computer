"""Stage 5 runner: sends questions to one arm's Strata server, one at a time, and saves everything per question
(appended to OUT/replies.jsonl as it goes, so a crash keeps what was done).

usage: python ask.py BASE_URL ARM QUESTIONS.jsonl OUT_DIR [--subset every4]
  ARM: A (no tools) | B (Python calc) | C (computer calc). B and C send "strata_mcp": true; A sends nothing extra.
  --subset every4: only primary questions (groups 1-4) whose position among them is a multiple of 4 (the fixed
  40-question determinism repeat of arm B).

Per question it saves: the request body, the full raw reply, wall seconds, the MCP server's call-log lines written
during the request (OUT/calls.jsonl), the engine-log lines written during it (OUT/strata.log), and harness errors.
Budgets (STAGE5-PLAN.md): max_tokens 8,192 per round, 20 tool rounds (server config), 600 s per question (on timeout
the connection is closed; Strata cancels a request whose client hung up).
Stop rules: any computer-vs-reference mismatch -> exit 3 (STOP file written); harness errors over 5% -> exit 4.
Two unscored warm-up questions run first; the first one's time is the arm's cold start.
Before each question, and after any failed one, it waits until GET /status reports nothing running or queued, so a
timed-out request's late tool calls are logged with that question and never leak into the next.
"""
import json, os, socket, sys, time, urllib.error, urllib.request

MODEL = "qwen3.8-flash-next-coder-iq1_m"
MAX_TOKENS, TIMEOUT = 8192, 600
WARMUP = ["Compute 123456789 * 987654321 exactly. End your reply with one line: ANSWER: <the exact value>.",
          "Compute 2 ** 100 exactly. End your reply with one line: ANSWER: <the exact value>."]
PRIMARY = {"1", "2", "3", "4"}


def body(prompt, arm):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
         "max_tokens": MAX_TOKENS, "stream": False}
    if arm in ("B", "C"):
        b["strata_mcp"] = True
    return b


def size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def tail_from(path, offset):
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def ask(base, b):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(b).encode(),
                                 {"Content-Type": "application/json"})
    t = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            d, code = json.load(r), r.status
    except urllib.error.HTTPError as e:
        d, code = {"error": e.read().decode(errors="replace")[:2000]}, e.code
    except (socket.timeout, TimeoutError):
        d, code = {"error": f"client timeout after {TIMEOUT} s (connection closed)"}, "timeout"
    except (urllib.error.URLError, OSError, ValueError) as e:
        d, code = {"error": f"{type(e).__name__}: {e}"}, None
    return code, d, round(time.monotonic() - t, 2)


def wait_idle(base, limit=300):
    """Wait until Strata reports nothing running or queued (GET /status). Returns the seconds waited, or None if the
    server never answered that it was idle within `limit` (or has no /status)."""
    t = time.monotonic()
    while time.monotonic() - t < limit:
        try:
            with urllib.request.urlopen(base + "/status", timeout=10) as r:
                st = json.load(r)
            if not st.get("busy") and not st.get("queued"):
                return round(time.monotonic() - t, 1)
        except (urllib.error.URLError, OSError, ValueError):
            return None
        time.sleep(1)
    return None


def main():
    base, arm, qpath, out = sys.argv[1:5]
    subset = "--subset" in sys.argv and sys.argv[sys.argv.index("--subset") + 1]
    os.makedirs(out, exist_ok=True)
    qs = [json.loads(l) for l in open(qpath) if l.strip()]
    if subset == "every4":
        prim = [q for q in qs if q["group"] in PRIMARY]
        qs = [q for i, q in enumerate(prim) if i % 4 == 0]
    calls_log, engine_log = os.path.join(out, "calls.jsonl"), os.path.join(out, "strata.log")
    rows_path = os.path.join(out, "replies.jsonl")
    items = [{"id": f"warmup-{i + 1}", "group": "warmup", "prompt": p} for i, p in enumerate(WARMUP)] + qs
    harness_errors = 0
    for n, q in enumerate(items):
        idle_before = wait_idle(base)                 # never start while an earlier request is still running
        c0, e0 = size(calls_log), size(engine_log)
        b = body(q["prompt"], arm)
        err = None
        try:
            code, d, secs = ask(base, b)
        except Exception as e:  # noqa: BLE001  (a bug in this script, not a server answer)
            code, d, secs, err = None, {}, None, f"{type(e).__name__}: {e}"
            harness_errors += 1
        idle_after = None
        if code != 200:                               # a timeout or an error: let the server finish or cancel first,
            idle_after = wait_idle(base)              # so late tool calls stay with this question, not the next
        calls = [json.loads(l) for l in tail_from(calls_log, c0).splitlines() if l.strip()]
        row = {"id": q["id"], "group": q["group"], "arm": arm, "request": b, "http": code, "seconds": secs,
               "raw": d, "tool_calls": calls, "engine_log": tail_from(engine_log, e0)[-20000:],
               "harness_error": err, "time": time.strftime("%X"),
               "idle_wait_before": idle_before, "idle_wait_after": idle_after}
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        c = (d.get("choices") or [{}])[0] if isinstance(d, dict) else {}
        print(f"{n + 1:3}/{len(items)} {q['id']:16} HTTP {code} {secs}s finish {c.get('finish_reason')} "
              f"calls {len(calls)}", flush=True)
        mism = [x for x in calls if str(x.get("error") or "").startswith("computer check failed")]
        if mism:
            open(os.path.join(out, "STOP"), "w").write(json.dumps({"id": q["id"], "calls": mism}) + "\n")
            print("STOP: the computer disagreed with its reference machine", flush=True)
            sys.exit(3)
        done = n + 1 - len(WARMUP)
        if done >= 10 and harness_errors / max(done, 1) > 0.05:
            print(f"STOP: harness errors {harness_errors} of {done}", flush=True)
            sys.exit(4)
    print(f"done: {len(items)} requests ({len(WARMUP)} warm-up), harness errors {harness_errors}")


if __name__ == "__main__":
    main()
