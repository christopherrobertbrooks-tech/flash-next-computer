"""Stage 4 bridge: an MCP server (stdio transport, newline-delimited JSON-RPC 2.0) offering one tool, `calc`, to
Strata's built-in MCP client (serve/mcp.py). Two interchangeable backends with the same interface:

  --backend computer  the exact computer (computer.py: Qwenputer's circuit on the V100, every token checked against
                      the reference machine; a fresh session per call, so calls are stateless like the Python tool)
  --backend python    upstream's exact Python calculator (calc.expected): the matched host-Python baseline and the
                      isolation control

Every call is logged twice to --log: a "begin" line when it arrives (with the question id the runner wrote into
--question-file) and an "end" line when it is answered (expression, result, error and its kind, timings, the process
id and its peak host RSS), including when the backend raises. Results go back as plain text; a refused or failed
call comes back with isError and text starting "error:".

usage (riscv dir + this dir on PYTHONPATH): python mcp_server.py --backend computer|python --model DIR --log FILE
"""
import argparse, json, os, resource, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DESCRIPTION = ("Evaluate an arithmetic expression exactly, with numbers of up to 62 digits: + - * / and parentheses "
               "on integers and fractions (exact: the result is an integer or a reduced fraction p/q), + - * on "
               "decimals, and on integers also // (floor division), %, ** and gcd(a,b). Returns the exact value.")
TOOL = {"name": "calc", "description": DESCRIPTION,
        "inputSchema": {"type": "object", "properties": {"expression": {
            "type": "string", "description": "e.g. 123456789*987654321-42 or 1.25*-3.5 or 84/18"}},
            "required": ["expression"]}}


def log_err(*a):
    print(*a, file=sys.stderr, flush=True)


def python_reference(calc, expr):
    """Upstream's exact Python value for an expression, read as the firmware reads input: recorded beside any result
    that says "error", so a bad expression (both say error) can be told from a computer limit (only the computer
    does) without asking anything at scoring time."""
    try:
        return calc.expected(str(expr).replace("\n", "").strip(" ")).strip()
    except Exception:  # noqa: BLE001  (no reference: the scorer then classifies the failure as "unknown")
        return None


class ComputerBackend:
    name = "computer"

    def __init__(self, model):
        from chipformer.rv import calc
        from computer import Computer
        firmware = Path(calc.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"
        self.calc = calc
        self.comp = Computer(model, firmware)

    def run(self, expr):
        """(result, error, info). info["error_kind"]: refused | budget | trace_mismatch | reference_error (from the
        service), or None. Only trace_mismatch means the computer and its reference machine disagreed."""
        from computer import CallRefused
        try:
            self.comp.new_session()
            r = self.comp.call(expr)
        except CallRefused as e:
            return None, f"refused: {e}", {"error_kind": "refused"}
        info = {"chip_tokens": r["chip_tokens"], "computer_seconds": r.get("seconds"), "error_kind": r["error_kind"]}
        if r["error"] is not None:
            return None, f"computer {r['error_kind']}: {r['error']}", info
        if r["result"].startswith("error"):
            info["python_reference"] = python_reference(self.calc, expr)
        return r["result"], None, info


class PythonBackend:
    """Upstream's exact calculator behind the same accepted-input contract as the computer (contract.py)."""
    name = "python"

    def __init__(self, model):
        from transformers import AutoTokenizer
        from chipformer.rv import bridge as BR, calc
        from contract import first_special_id
        self.calc = calc
        self.tok = AutoTokenizer.from_pretrained(model)
        self.T = BR.Text(Path(model) / "tokenizer.json")
        self.first_special = first_special_id(self.tok)

    def run(self, expr):
        from contract import CallRefused, check_input
        try:
            check_input(expr, self.tok, self.T, self.first_special)
        except CallRefused as e:
            return None, f"refused: {e}", {"error_kind": "refused"}
        expr = expr.replace("\n", "").strip(" ")   # read the input as the firmware does: newlines dropped, spaces skipped
        try:
            result = self.calc.expected(expr).strip()
        except Exception as e:  # noqa: BLE001
            return None, f"python: {type(e).__name__}: {e}", {"error_kind": "execution_error"}
        info = {"error_kind": None}
        if result.startswith("error"):
            info["python_reference"] = result
        return result, None, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["computer", "python"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--question-file", help="a file the runner writes the current question id into; each call is "
                    "logged with the id it held when the call arrived")
    a = ap.parse_args()
    t = time.time()
    backend = (ComputerBackend if a.backend == "computer" else PythonBackend)(a.model)
    log_err(f"[calc-mcp] {a.backend} backend ready in {time.time() - t:.1f} s")
    logf = open(a.log, "a", buffering=1)
    out = sys.stdout

    instance = f"{os.getpid()}-{time.time_ns()}"     # Strata restarts a server with RPC ids from 0 again: calls are
                                                     # matched on (instance, rpc_id), never rpc_id alone

    def log(rec):
        logf.write(json.dumps({"instance": instance, **rec}) + "\n")

    def resources():
        return {"pid": os.getpid(), "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}

    def send(msg):
        out.write(json.dumps(msg) + "\n")
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:                                    # a notification (notifications/initialized, cancelled)
            continue
        try:
            if method == "initialize":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "protocolVersion": (msg.get("params") or {}).get("protocolVersion", "2025-06-18"),
                    "capabilities": {"tools": {}}, "serverInfo": {"name": f"calc-{a.backend}", "version": "0.1"}}})
            elif method == "ping":
                send({"jsonrpc": "2.0", "id": mid, "result": {}})
            elif method == "tools/list":
                send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [TOOL]}})
            elif method == "tools/call":
                p = msg.get("params") or {}
                expr = (p.get("arguments") or {}).get("expression")
                question = None
                if a.question_file:
                    try:
                        question = Path(a.question_file).read_text().strip()
                    except OSError:
                        question = "(unreadable)"
                t0 = time.time()
                log({"phase": "begin", "time": time.strftime("%X"), "rpc_id": mid, "question": question,
                     "expression": expr})
                try:
                    if p.get("name") != "calc":
                        result, err, info = None, f"unknown tool {p.get('name')!r}", {"error_kind": "unknown_tool"}
                    else:
                        result, err, info = backend.run(expr)
                except Exception as e:  # noqa: BLE001  (logged, then answered as an RPC error below)
                    log({"phase": "end", "time": time.strftime("%X"), "rpc_id": mid, "question": question,
                         "backend": a.backend, "expression": expr, "result": None,
                         "error": f"{type(e).__name__}: {e}", "error_kind": "execution_error",
                         "ms": round((time.time() - t0) * 1000, 1), **resources()})
                    raise
                log({"phase": "end", "time": time.strftime("%X"), "rpc_id": mid, "question": question,
                     "backend": a.backend, "expression": expr, "result": result, "error": err,
                     "ms": round((time.time() - t0) * 1000, 1), **info, **resources()})
                text = result if err is None else f"error: {err}"
                send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}],
                                                              "isError": err is not None}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}})
        except Exception as e:  # noqa: BLE001
            log_err(traceback.format_exc())
            send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}})


if __name__ == "__main__":
    main()
