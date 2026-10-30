"""CPU-only controls for ask.py against a fake Strata (no model, no GPU). Exits 1 if any control fails.

The fake mimics the two things that matter for attribution: requests run one at a time (FIFO), and GET /status says
idle while a tool call can still be running (Strata 0.1.41 runs MCP tools after the generation round ends). Its tool
calls are tagged from the runner's question file, like computer/mcp_server.py.

Controls:
  1. a normal question: its call is attributed to it;
  2. HTTP 503: recorded, no calls;
  3. a timeout whose tool call is still running when /status says idle, and ends 4 s later: the late call must stay
     with the timed-out question, and the next question must not start until it ended;
  4. a trace mismatch: the arm stops (exit 3) on the right question;
  5. a server without /status: the runner refuses to go on (exit 5, fail closed);
  6. a budget error from the computer (not a mismatch): recorded, the arm goes on.
"""
import json, os, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ask  # noqa: E402


def fake_server(out, port, with_status=True):
    fifo, state = threading.Lock(), {"busy": 0}
    calls = os.path.join(out, "calls.jsonl")
    rpc = {"n": 0}

    def log(rec):
        with open(calls, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def tool_call(expr, result=None, err=None, kind=None, delay=0.0):
        rpc["n"] += 1
        mid = rpc["n"]
        tag = Path(out, "current_question").read_text().strip()
        log({"phase": "begin", "rpc_id": mid, "question": tag, "expression": expr})
        if delay:
            time.sleep(delay)
        log({"phase": "end", "rpc_id": mid, "question": tag, "expression": expr, "result": result, "error": err,
             "error_kind": kind, "ms": 1.0})

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if not with_status:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"busy": state["busy"] > 0, "queued": 0}).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["messages"][0]["content"]
            with fifo:
                state["busy"] += 1
                try:
                    if "FAIL" in prompt:
                        self.send_response(503)
                        self.end_headers()
                        self.wfile.write(b'{"error": "engine stopped"}')
                        return
                    if "SLOW" in prompt:          # the tool runs on after /status says idle
                        threading.Thread(target=tool_call, args=("9*9", "81"), kwargs={"delay": 4.0}).start()
                        state["busy"] -= 1
                        time.sleep(3)
                        state["busy"] += 1
                    elif "MISMATCH" in prompt:
                        tool_call("1+1", None, "computer trace_mismatch: {'at': 3}", "trace_mismatch")
                    elif "BUDGET" in prompt:
                        tool_call("2**999", None, "computer budget: more than 120 s", "budget")
                    elif body.get("strata_mcp"):
                        tool_call("2+2", "4")
                    d = {"choices": [{"finish_reason": "stop", "message": {"content": "ANSWER: 4"}}]}
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(d).encode())
                    except BrokenPipeError:
                        pass
                finally:
                    state["busy"] -= 1

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def run(out, port, prompts):
    q = os.path.join(out, "q.jsonl")
    with open(q, "w") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"id": f"q{i + 1}", "group": "1", "prompt": p}) + "\n")
    sys.argv = ["ask.py", f"http://127.0.0.1:{port}", "C", q, out]
    try:
        ask.main()
        code = 0
    except SystemExit as e:
        code = e.code
    rows = [json.loads(l) for l in open(os.path.join(out, "replies.jsonl"))]
    return code, {r["id"]: r for r in rows}


def main():
    ask.TIMEOUT = 2
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    with tempfile.TemporaryDirectory() as out:
        srv = fake_server(out, 18101)
        code, rows = run(out, 18101, ["normal", "FAIL", "SLOW", "normal", "BUDGET", "MISMATCH", "never sent"])
        srv.shutdown()
        exprs = {k: [c["expression"] for c in r.get("tool_calls") or []] for k, r in rows.items()}
        check("normal question keeps its call", exprs["q1"] == ["2+2"])
        check("HTTP 503 recorded with no calls", rows["q2"]["http"] == 503 and exprs["q2"] == [])
        check("timed-out question keeps its late call", rows["q3"]["http"] == "timeout" and exprs["q3"] == ["9*9"])
        check("the next question gets only its own call", exprs["q4"] == ["2+2"])
        check("the next question started only after the late call ended",
              rows["q3"].get("fence_after") is not None and rows["q3"]["fence_after"] >= 1)
        check("a budget error is recorded and the arm goes on",
              rows["q5"]["tool_calls"][0]["error_kind"] == "budget" and "q6" in rows)
        check("a trace mismatch stops the arm on the right question (exit 3)",
              code == 3 and json.load(open(os.path.join(out, "STOP")))["id"] == "q6" and "q7" not in rows)
    with tempfile.TemporaryDirectory() as out:
        srv = fake_server(out, 18102, with_status=False)
        code, rows = run(out, 18102, ["normal", "normal"])
        srv.shutdown()
        check("no /status: the arm stops before sending anything (exit 5)",
              code == 5 and rows["warmup-1"].get("request") is None and len(rows) == 1)
    print("FAKE-SERVER CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
