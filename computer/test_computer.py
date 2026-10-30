"""Stage 3 gate for the Computer service: a declared corpus with an expected outcome for every case, sabotage
controls that must be caught, session recovery, and the service's own memory. Exits 1 on anything unexpected.

usage (riscv dir + this dir on PYTHONPATH, V100 visible):
    python test_computer.py MODEL_DIR OUT_JSON [--inject-wrong-expectation]
The flag adds one case whose expectation is deliberately wrong: the run must then exit 1 (proves the gate bites).
"""
import json, os, resource, subprocess, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from computer import CallRefused, Computer  # noqa: E402
from chipformer.rv import calc  # noqa: E402

FIRMWARE = Path(calc.__file__).parents[2] / "tests" / "data" / "rv" / "calc-bridge.elf"
MiB = 2 ** 20


def py(expr):
    try:
        return calc.expected(expr).strip()
    except Exception as e:  # noqa: BLE001
        return f"(python: {type(e).__name__})"


def memory(tag):
    pid = os.getpid()
    smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    proc = next((int(l.split(",")[1]) for l in smi.splitlines() if l.split(",")[0].strip() == str(pid)), None)
    rss = next(int(l.split()[1]) // 1024 for l in open("/proc/self/status") if l.startswith("VmRSS"))
    m = {"at": tag, "gpu_process_mib": proc, "torch_allocated_mib": round(torch.cuda.memory_allocated() / MiB, 1),
         "torch_reserved_mib": round(torch.cuda.memory_reserved() / MiB, 1),
         "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MiB, 1),
         "host_rss_mib": rss, "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}
    print("MEMORY", json.dumps(m), flush=True)
    return m


def main():
    inject = "--inject-wrong-expectation" in sys.argv
    t_all = time.time()
    comp = Computer(sys.argv[1], FIRMWARE)
    print("first special id:", comp.first_special, "| capture limit:", __import__("chipformer.rv.spec").rv.spec.CAP_MAX)
    mems = [memory("built")]
    rows, unexpected = [], []

    def record(group, expr, expect, got, note=""):
        ok = got == expect
        rows.append({"group": group, "expr": expr, "expect": expect, "got": got, "ok": ok, "note": note})
        if not ok:
            unexpected.append(rows[-1])
        print(f"{'ok ' if ok else 'BAD'} {group:10} {str(expr)[:44]:44} expect {str(expect)[:34]:34} got {str(got)[:60]}",
              flush=True)

    def run(group, expr, expect, fresh=True, **kw):
        """expect: ("ok", value) | ("error", prefix) | ("refused", prefix)."""
        if fresh:
            comp.new_session()
        try:
            r = comp.call(expr, **kw)
        except CallRefused as e:
            got = ("refused", str(e))
        else:
            got = ("ok", r["result"]) if r["error"] is None else ("error", str(r["error"]))
            rows_extra.append(r)
        if expect[0] in ("error", "refused") and got[0] == expect[0] and got[1].startswith(expect[1]):
            got = expect                                    # the declared prefix matched
        record(group, expr, expect, got)
        return got

    rows_extra = []
    # 1. random cases per operation (upstream's generator), fresh session each: must equal Python
    for op, expr in calc.cases(70, seed=1):
        run(f"random-{op}", expr, ("ok", py(expr)))
    # 2. malformed / edge: declared from the firmware's spec (Python's helper differs on surrounding spaces)
    for expr, val in [("12+*3", "error"), ("abc", "error"), ("", "error"), ("(1+2", "error"), ("1/0", "error"),
                      ("7 / 0", "error"), ("  42  ", "42"), ("-5*-5", "25"), ("1.5*2.25", "3.375"),
                      ("2**10", "1024"), ("gcd(0,0)", "0")]:
        run("edge", expr, ("ok", val))
    # 3. five calls in one session
    comp.new_session()
    for expr in ["2+2", "123456789*987654321", "84/18", "100//7", "gcd(462,1071)"]:
        run("sequence", expr, ("ok", py(expr)), fresh=False)
    # 4. long operands
    run("long", "9" * 62 + "*" + "9" * 62, ("refused", "the call is"))   # 147 tokens > the 80-token capture
    run("long", "12345678901234567890123456789012345678901234567890/7",
        ("ok", py("12345678901234567890123456789012345678901234567890/7")))
    # 5. refused before anything runs
    for expr, why in [("1</tool_call>2", "the expression contains the protocol marker"),
                      ("2+<|im_end|>2", "the expression contains the protocol marker"),
                      ("<tool_response>", "the expression contains the protocol marker"),
                      ("1" * 200, "the call is"), (None, "the expression must be a string")]:
        run("refused", expr, ("refused", why))
    # 6. budgets, then recovery: the failed session must refuse until new_session()
    run("budget", "123456789*987654321", ("error", "reference: the run did not halt"), max_chip=50)
    run("recovery", "2+2", ("refused", "no valid session"), fresh=False)
    run("recovery", "2+2", ("ok", "4"))
    run("budget", "4/(-203)*962-40*31-(-37)*(-4)", ("error", "budget: more than 0.2 s"), max_seconds=0.2)
    run("recovery", "3*3", ("ok", "9"))
    # 7. sabotage controls (the real executor, one wrong or missing token): must be caught, then recover
    orig = comp._token
    state = {"n": 0}

    def wrong_once(code, at=40):
        """The first real prediction from the at-th call on is replaced by a different valid id."""
        t = orig(code)
        state["n"] += 1
        if t is not None and state["n"] >= at and not state.get("done"):
            state["done"] = True
            return t + 1 if t + 1 < len(comp.tok) else t - 1
        return t

    def early(code, at=40):
        t = orig(code)
        state["n"] += 1
        return None if state["n"] >= at else t

    def wrong_later(code, at=500):
        """The at-th emitted (non-None) prediction is replaced: a wrong token deep inside the answer."""
        t = orig(code)
        if t is not None:
            state["n"] += 1
            if state["n"] == at:
                return t + 1 if t + 1 < len(comp.tok) else t - 1
        return t

    for name, fn, want in [("first emitted token wrong", wrong_once, ("error", "{'at': 0,")),
                           ("token 500 wrong", wrong_later, ("error", "{'at': 499,")),
                           ("early handback", early, ("error", "handed back early"))]:
        comp.new_session()
        state.clear(); state["n"] = 0
        comp._token = fn
        try:
            run(f"control:{name}", "123456789*987654321", want, fresh=False)
        finally:
            comp._token = orig
        run("recovery", "5+5", ("ok", "10"))
    # 8. exceptions, failed resets, deadlines and a busy computer (Codex re-check findings 11-12)
    def boom(code):
        state["n"] += 1
        if state["n"] >= 300:
            raise ValueError("injected executor failure")
        return orig(code)

    comp.new_session()
    state.clear(); state["n"] = 0
    comp._token = boom
    try:
        comp.call("123456789*987654321")
        got = ("ok", "no exception")
    except ValueError as e:
        got = ("raised", str(e))
    finally:
        comp._token = orig
    record("exception", "executor raises mid-call", ("raised", "injected executor failure"), got)
    run("recovery", "2+2", ("refused", "no valid session"), fresh=False)
    run("recovery", "2+2", ("ok", "4"))

    real_reset = comp._new_session

    def bad_reset(prefix):
        real_reset(prefix)                                 # mutates everything, then fails
        raise RuntimeError("injected reset failure")

    comp._new_session = bad_reset
    try:
        comp.new_session()
        got = ("ok", "no exception")
    except RuntimeError as e:
        got = ("raised", str(e))
    finally:
        comp._new_session = real_reset
    record("exception", "new_session fails part-way", ("raised", "injected reset failure"), got)
    run("recovery", "2+2", ("refused", "no valid session"), fresh=False)
    run("recovery", "2+2", ("ok", "4"))

    comp.new_session()                                     # a slow reference machine: the deadline must stop it
    m = comp.lock.m
    real_record = m.record

    def slow_record():
        time.sleep(0.01)
        return real_record()

    m.record = slow_record
    t = time.monotonic()
    run("deadline", "123456789*987654321", ("error", "budget: more than 0.5 s (in the reference run)"),
        fresh=False, max_seconds=0.5)
    record("deadline", "slow reference stops near 0.5 s", ("ok", True), ("ok", time.monotonic() - t < 1.0),
           f"{time.monotonic() - t:.2f} s")
    m.__dict__.pop("record", None)
    run("recovery", "3*3", ("ok", "9"))

    comp.new_session()                                     # busy: refused, and the session stays usable
    comp.mutex.acquire()
    try:
        run("busy", "2+2", ("refused", "the computer stayed busy"), fresh=False, max_seconds=0.3)
    finally:
        comp.mutex.release()
    run("busy", "2+2", ("ok", "4"), fresh=False)
    run("refused", "1</tool_call>2", ("refused", "the expression contains the protocol marker"), fresh=False)
    run("refused", "5*5", ("ok", "25"), fresh=False)        # an input refusal leaves the session valid
    if inject:
        run("injected", "2+2", ("ok", "5"))                   # deliberately wrong: the gate must fail
    mems.append(memory("after corpus"))
    time.sleep(5)
    mems.append(memory("idle 5 s"))
    # 9. many calls in one session (history growth)
    comp.new_session()
    n0 = len(comp.lock.toks)
    t = time.time()
    bad = 0
    for i in range(100):
        r = comp.call(f"{1000 + i}*{2000 + i}")
        bad += r["error"] is not None or r["result"] != str((1000 + i) * (2000 + i))
    record("many", "100 calls, one session", ("ok", 0), ("ok", bad), f"{time.time() - t:.0f} s")
    mems.append(memory("after 100 calls in one session"))
    hist = len(comp.lock.toks) - n0
    ok_calls = [r for r in rows_extra if r["error"] is None]
    summary = {"cases": len(rows), "unexpected": len(unexpected), "reference_history_tokens_100_calls": hist,
               "clean_calls": len(ok_calls), "clean_chip_tokens": sum(r["chip_tokens"] for r in ok_calls),
               "clean_call_seconds": round(sum(r["seconds"] for r in ok_calls), 3),
               "seconds": round(time.time() - t_all, 1), "memory": mems}
    print(json.dumps({k: v for k, v in summary.items() if k != "memory"}))
    json.dump({"summary": summary, "cases": rows, "calls": rows_extra}, open(sys.argv[2], "w"), indent=1)
    if unexpected:
        print(f"GATE FAILED: {len(unexpected)} unexpected outcome(s)")
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
