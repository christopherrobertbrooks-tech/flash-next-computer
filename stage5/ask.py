"""Stage 5 runner: sends questions to one arm's Strata server, one at a time, and saves everything per question
(appended to OUT/replies.jsonl as it goes, so a crash keeps what was done).

usage: python ask.py BASE_URL ARM QUESTIONS.jsonl OUT_DIR [--subset every4]
  ARM: A (no tools) | B (Python calc) | C (computer calc). B and C send "strata_mcp": true; A sends nothing extra.
  --subset every4: only primary questions (groups 1-4) whose position among them is a multiple of 4 (the fixed
  40-question determinism repeat of arm B).

Per question it saves: the request body, the full raw reply, wall seconds, its tool calls (by tag, from
OUT/calls.jsonl), the engine-log lines written during it (OUT/strata.log, untruncated, with byte offsets), and any
harness error.
Budgets (STAGE5-PLAN.md): max_tokens 8,192 per round, 20 tool rounds (server config), 600 s per question (on timeout
the connection is closed; Strata cancels a request whose client hung up).
Stop rules: a computer-vs-reference trace mismatch (error_kind trace_mismatch) -> exit 3; harness errors over
5% -> exit 4; the arm not provably quiet -> exit 5. Each writes OUT/STOP.
Two unscored warm-up questions run first; the first one's time is the arm's cold start.
Fencing: the runner writes the current question id into OUT/current_question; the tool server tags each call with it
when the call arrives and logs a begin and an end line. Calls are attributed by that tag, never by log position.
Before each question the arm must be quiet (GET /status idle, no call begun-but-unended); after a failed request it
must also stay quiet for 10 s. If that can't be shown within 300 s, the attempt is saved and the arm stops (exit 5):
fail closed. Every attempt is saved, including local failures, and counts toward the 5% harness rule.
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


def read_calls(path):
    """(begin lines, end lines, problems) of the MCP call log. Calls are matched one-to-one on (instance, rpc_id):
    the tool server's instance id changes when Strata restarts it, and RPC ids start again. A line that isn't JSON,
    a duplicate key, or an end with no begin is a problem (a harness error for the question)."""
    begins, ends, problems = [], [], []
    for l in tail_from(path, 0).splitlines():
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except ValueError:
            problems.append(f"unreadable line {l[:80]!r}")
            continue
        (begins if r.get("phase") == "begin" else ends).append(r)
    key = lambda r: (r.get("instance"), r.get("rpc_id"))  # noqa: E731
    for name, rows in (("begin", begins), ("end", ends)):
        keys = [key(r) for r in rows]
        dup = {k for k in keys if keys.count(k) > 1}
        if dup:
            problems.append(f"duplicate {name} keys {sorted(dup, key=str)[:3]}")
        if any(k[0] is None for k in keys):
            problems.append(f"{name} lines without an instance id")
    begun = {key(r) for r in begins}
    orphans = [key(r) for r in ends if key(r) not in begun]
    if orphans:
        problems.append(f"ends with no begin {orphans[:3]}")
    return begins, ends, problems


def open_calls(path):
    """Calls that began and have not ended, matched on (instance, rpc_id)."""
    begins, ends, _ = read_calls(path)
    done = {(r.get("instance"), r.get("rpc_id")) for r in ends}
    return [b for b in begins if (b.get("instance"), b.get("rpc_id")) not in done]


FENCE_LIMIT = 300


def fence(base, calls_log, quiet=10, limit=None):
    """Proves the arm is quiet before the next question: Strata reports nothing running or queued (GET /status), every
    tool call that began has ended, and nothing new was logged for `quiet` seconds. Returns seconds waited, or None
    if that couldn't be shown within `limit` (the caller then stops the arm: fail closed)."""
    limit = FENCE_LIMIT if limit is None else limit
    t = time.monotonic()
    last_change, last_size = time.monotonic(), size(calls_log)
    while time.monotonic() - t < limit:
        idle = wait_idle(base, limit=max(1, limit - (time.monotonic() - t)))
        if idle is None:
            return None
        if size(calls_log) != last_size:
            last_size, last_change = size(calls_log), time.monotonic()
        if not open_calls(calls_log) and time.monotonic() - last_change >= quiet:
            return round(time.monotonic() - t, 1)
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
    marker = os.path.join(out, "current_question")
    rows_path = os.path.join(out, "replies.jsonl")
    items = [{"id": f"warmup-{i + 1}", "group": "warmup", "prompt": p} for i, p in enumerate(WARMUP)] + qs
    harness_errors = 0
    for n, q in enumerate(items):
        row = {"id": q["id"], "group": q["group"], "arm": arm, "time": time.strftime("%X"), "harness_error": None}
        try:
            row["fence_before"] = fence(base, calls_log, quiet=0)   # nothing may still run from earlier questions
            if row["fence_before"] is None:
                raise RuntimeError("the arm was not quiet before this question")
            open(marker, "w").write(q["id"])           # tool calls arriving from now on carry this question's id
            c0, e0 = size(calls_log), size(engine_log)
            b = body(q["prompt"], arm)
            row["request"] = b
            code, d, secs = ask(base, b)
            row.update({"http": code, "seconds": secs, "raw": d})
            open(marker, "w").write("after:" + q["id"])  # stragglers are still this question's, by their tag
            mine = {q["id"], "after:" + q["id"]}
            if code != 200 or any(b.get("question") in mine for b in open_calls(calls_log)):
                row["fence_after"] = fence(base, calls_log)   # a failed request, or one of its calls still running:
                if row["fence_after"] is None:                # wait until everything has ended (fail closed)
                    raise RuntimeError("could not prove the arm quiet after the request")
            row["engine_log_offsets"] = [e0, size(engine_log)]
            row["engine_log"] = tail_from(engine_log, e0)
            begins, ends, bad = read_calls(calls_log)
            mine = {q["id"], "after:" + q["id"]}
            done = {(e.get("instance"), e.get("rpc_id")) for e in ends}
            row["tool_calls"] = [r for r in ends if r.get("question") in mine]
            row["tool_calls_open"] = [r for r in begins if r.get("question") in mine and
                                      (r.get("instance"), r.get("rpc_id")) not in done]
            row["calls_log_offsets"] = [c0, size(calls_log)]
            stray = [r for r in ends if r.get("question") not in {x["id"] for x in items[:n + 1]} |
                     {"after:" + x["id"] for x in items[:n + 1]}]
            if bad or stray:
                raise RuntimeError(f"call log problems: {bad[:3]}; {len(stray)} calls with an unknown question tag")
        except Exception as e:  # noqa: BLE001  (this script or the fence failed: keep the attempt, stop if needed)
            row["harness_error"] = f"{type(e).__name__}: {e}"
            harness_errors += 1
            try:                                          # keep what the call log shows, as evidence
                begins, ends, _ = read_calls(calls_log)
                mine = {q["id"], "after:" + q["id"]}
                done = {(x.get("instance"), x.get("rpc_id")) for x in ends}
                row.setdefault("tool_calls", [r for r in ends if r.get("question") in mine])
                row.setdefault("tool_calls_open", [r for r in begins if r.get("question") in mine and
                                                   (r.get("instance"), r.get("rpc_id")) not in done])
            except Exception:  # noqa: BLE001
                pass
        with open(rows_path, "a") as f:                   # every attempt is saved, whatever happened
            f.write(json.dumps(row) + "\n")
        d = row.get("raw") or {}
        c = (d.get("choices") or [{}])[0] if isinstance(d, dict) else {}
        print(f"{n + 1:3}/{len(items)} {q['id']:16} HTTP {row.get('http')} {row.get('seconds')}s finish "
              f"{c.get('finish_reason')} calls {len(row.get('tool_calls') or [])}"
              f"{' HARNESS: ' + row['harness_error'] if row['harness_error'] else ''}", flush=True)
        mism = [x for x in row.get("tool_calls") or [] if x.get("error_kind") == "trace_mismatch"]
        if mism:
            open(os.path.join(out, "STOP"), "w").write(json.dumps({"id": q["id"], "calls": mism}) + "\n")
            print("STOP: the computer disagreed with its reference machine", flush=True)
            sys.exit(3)
        if row["harness_error"] and ("quiet" in row["harness_error"]):
            open(os.path.join(out, "STOP"), "w").write(json.dumps({"id": q["id"], "why": row["harness_error"]}) + "\n")
            print("STOP: the arm could not be proven quiet (fail closed)", flush=True)
            sys.exit(5)
        done = n + 1
        if done >= 10 and harness_errors / done > 0.05:
            print(f"STOP: harness errors {harness_errors} of {done}", flush=True)
            sys.exit(4)
    print(f"done: {len(items)} requests ({len(WARMUP)} warm-up), harness errors {harness_errors}")


if __name__ == "__main__":
    main()
