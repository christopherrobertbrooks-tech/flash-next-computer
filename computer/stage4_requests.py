"""Stage 4 requests: fixed prompts, temperature 0, MCP on, non-streaming; saves every reply.
usage: python stage4_requests.py BASE_URL OUT_JSON"""
import json, sys, time, urllib.request

PROMPTS = [
    ("mul", "What is 123456789 * 987654321? Use the calc tool, then give the exact answer."),
    ("pow-floordiv", "Compute (2**64 - 1) // 7 exactly. Use the calc tool."),
    ("two-step", "Multiply 98765432123 by 12345678987, then divide that product by 7. Use the calc tool for each "
                 "step and give the exact final value as a reduced fraction."),
    ("no-tool", "Write a three-line haiku about rain. Don't use any tools."),
]


def ask(base, text):
    body = {"model": "qwen3.8-flash-next-coder-iq1_m", "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": 3000, "stream": False, "strata_mcp": True}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
            code = r.status
    except urllib.error.HTTPError as e:
        d, code = {"error": e.read().decode()[:500]}, e.code
    except (urllib.error.URLError, OSError, ValueError) as e:     # no answer, or not JSON
        d, code = {"error": f"{type(e).__name__}: {e}"}, None
    return code, d, round(time.time() - t, 1)


def main():
    base, out = sys.argv[1:3]
    rows = []
    for name, text in PROMPTS:
        code, d, s = ask(base, text)
        c = (d.get("choices") or [{}])[0]
        m = c.get("message") or {}
        row = {"name": name, "http": code, "seconds": s, "finish": c.get("finish_reason"), "usage": d.get("usage"),
               "content": m.get("content"), "reasoning_chars": len(m.get("reasoning_content") or ""),
               "raw_keys": sorted(d)}
        rows.append(row)
        print(f"{name:13} HTTP {code} {s:6.1f}s finish {row['finish']} usage {row['usage']}\n   content: "
              f"{(row['content'] or '')[:220]!r}", flush=True)
    json.dump({"replies": rows, "raw_last": d}, open(out, "w"), indent=1)
    bad = [r["name"] for r in rows if r["http"] != 200 or r["finish"] is None]
    if bad:
        print("BAD REPLIES:", ", ".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
