"""Stage 4 bridge: an MCP server (stdio transport, newline-delimited JSON-RPC 2.0) offering one tool, `calc`, to
Strata's built-in MCP client (serve/mcp.py). Two interchangeable backends with the same interface:

  --backend computer  the exact computer (computer.py: Qwenputer's circuit on the V100, every token checked against
                      the reference machine; a fresh session per call, so calls are stateless like the Python tool)
  --backend python    upstream's exact Python calculator (calc.expected): the matched host-Python baseline and the
                      isolation control

Every call is appended to --log as one JSON line (expression, backend, result, error, timings). Results go back as
plain text; a refused or failed call comes back with isError and text starting "error:".

usage (riscv dir + this dir on PYTHONPATH): python mcp_server.py --backend computer|python --model DIR --log FILE
"""
import argparse, json, sys, time, traceback
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


class ComputerBackend:
    name = "computer"

    def __init__(self, model):
        from chipformer.rv import calc
        from computer import Computer
        firmware = Path(calc.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"
        self.comp = Computer(model, firmware)

    def run(self, expr):
        from computer import CallRefused
        try:
            self.comp.new_session()
            r = self.comp.call(expr)
        except CallRefused as e:
            return None, f"refused: {e}", {}
        info = {"chip_tokens": r["chip_tokens"], "computer_seconds": r.get("seconds")}
        if r["error"] is not None:
            return None, f"computer check failed: {r['error']}", info
        return r["result"], None, info


class PythonBackend:
    name = "python"

    def __init__(self, model):
        from chipformer.rv import calc
        self.calc = calc

    def run(self, expr):
        if not isinstance(expr, str):
            return None, "refused: the expression must be a string", {}
        try:
            return self.calc.expected(expr).strip(), None, {}
        except Exception as e:  # noqa: BLE001
            return None, f"python: {type(e).__name__}: {e}", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["computer", "python"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    t = time.time()
    backend = (ComputerBackend if a.backend == "computer" else PythonBackend)(a.model)
    log_err(f"[calc-mcp] {a.backend} backend ready in {time.time() - t:.1f} s")
    logf = open(a.log, "a", buffering=1)
    out = sys.stdout

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
                t0 = time.time()
                if p.get("name") != "calc":
                    result, err, info = None, f"unknown tool {p.get('name')!r}", {}
                else:
                    result, err, info = backend.run(expr)
                ms = round((time.time() - t0) * 1000, 1)
                logf.write(json.dumps({"time": time.strftime("%X"), "backend": a.backend, "expression": expr,
                                       "result": result, "error": err, "ms": ms, "rpc_id": mid, **info}) + "\n")
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
