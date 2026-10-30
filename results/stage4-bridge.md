# Stage 4, first proof: Flash-Next calls the computer through Strata's MCP tools (2026-10-30)

**Setup:** Strata 0.1.41, its own launch on port 8091 (llama-swap's live config untouched). The novision config was
used, plus one MCP server, `computer/mcp_server.py`, offering one tool, `calc`. Script: `scripts/stage4_mcp_test.sh`.
The same server has two backends with the same tool description and the same result text:
- `computer`: the checked computer, with a fresh session for every call;
- `python`: upstream's exact Python calculator (`calc.expected`).

Four fixed prompts were sent (`computer/stage4_requests.py`) at temperature 0, non-streaming, with `"strata_mcp": true`:
a multiplication, `(2**64-1)//7`, a two-step product then division, and a haiku with no tools. Each run was a fresh
Strata launch inside the V100 lock. Raw outputs are in `stage4-runs/`.

## Results

| Run | Tool calls | Exact | Words Flash-Next wrote | Request times (s) | V100 peak |
|---|---|---|---|---|---|
| computer-1 | 4 | 4/4 | (first try; same prompts) | | 32,417 MiB |
| computer-2 | 4 | 4/4 | identical to python-1 | 10.3, 8.5, 16.6, 23.8 | 32,417 MiB |
| python-1 | 4 | 4/4 | identical to python-2 | 3.3, 5.4, 11.9, 23.8 | 31,987 MiB |
| python-2 | 4 | 4/4 | identical to python-1 | 3.3, 5.4, 11.9, 23.8 | 31,987 MiB |

- **Flash-Next called the tool by itself**, with the right expressions, every time. The computer returned
  `121932631112635269`, `2635249153387078802`, `1219326320002895899401` and the reduced fraction
  `1219326320002895899401/7`, all equal to Python.
- **Same outputs either way, on these four prompts.** With the computer and with Python, the saved final reply text,
  finish reason, token counts and reasoning length were identical for all four prompts, and Python run twice was
  identical too. The bridge design keeps the computer's internal trace tokens in its own process; whether Strata's
  internal state (GDN, KV, indexer, MTP) is byte-identical, and whether behaviour matches on other inputs, is **not**
  shown by this and remains open (pause/resume isolation).
- **Prefix reuse works after each tool round.** For example, "prompt 557 tokens = 526 reused" after the first call:
  Strata did not re-read the conversation.
- **Cost: time.** Each computer call took 1.4-3.2 s. Warm Python calls were logged at 0.1 ms resolution; first calls
  took 3.0-3.7 ms. Python was much faster on every observed call; exact warm ratios are limited by the timing
  resolution, and end-to-end task latency is the comparison that matters. The first computer request was also
  about 5 s slower than the call itself accounts for. Most likely it waited for the computer to finish loading (Strata
  starts the MCP server in the background after it says "ready"); this is not yet measured separately.
- **Memory:** the computer process adds about 430 MiB. With all 12,288 experts resident, the V100 peaked at
  32,417 of 32,768 MiB: it fits, with about 350 MiB to spare. The script tried to fix the expert cache at 12,000
  slots, but `--resident-experts` overrides `--expert-cache` (the log still says 12,288 slots). The override was
  removed and the script now says so.
- The haiku prompt ran out of tokens while thinking (2,560 tokens, all thinking; `finish: length`, empty answer). It does
  this with both backends and never calls a tool, so this is Flash-Next's thinking, not the bridge.

## What this does and doesn't show
It shows the bridge works end to end on these prompts: Flash-Next on the V100 calls an exact computer and gets checked,
correct answers, with the same final outputs as with Python. It does **not** show state isolation (above) or
usefulness: on these prompts the computer and Python give identical outcomes, and Python is far faster. Stage 5 measures whether anything differs on a real task set
(Flash-Next alone vs with Python vs with the computer).

Still open from Codex's stage 4 list: pause/resume isolation, cancellation, two consecutive conversations, MTP on/off,
end-to-end latency on more prompts, vision + service memory, and a frozen inputs manifest.

## Codex re-check of eb76149 (3443e87) and the fixes
- **15 (redline): the driver passed wrong answers.** It checked only HTTP and a finish reason, so `RUN OK` meant
  "transport worked". Fixed: `computer/stage4_requests.py` now declares an expectation per prompt (a tool call, the
  exact tool results, no tool errors, `finish: stop`, the exact final value in the answer with digit boundaries; for
  the haiku, zero tool calls, and its finish recorded separately as "haiku completed" or "known: ran out while
  thinking") and exits 1 unless all are met. `--selftest` (no server) runs 15 cases: 4 good shapes pass, 11 bad ones
  (wrong answer, no tool, tool error, wrong tool result, missing second step, tool used for the haiku, HTTP 500, no
  finish, cut-off answer, a longer number containing the right digits) all fail; a deliberately broken copy of the
  self-test exits 1. The four saved runs re-checked with it: all expectations met (tool calls split by their order
  in the log, since those runs predate per-request saving).
- **16 (redline): "no trace" overclaimed.** Reworded above.
- **17: not every raw reply was saved.** Fixed: each request row now holds the request body, the full raw reply, its
  tool calls and the verdict; the MCP log records each call's JSON-RPC id. The four runs above have summaries and only
  the last raw reply.
- **18: the speed ratio ignored cold Python calls.** Reworded above.
- **Not claimed:** cancellation (the MCP server handles one request at a time and ignores cancel notifications).
  **Before stage 5:** the Python backend calls `calc.expected` directly, while the computer refuses markers, special
  tokens and calls over the 80-token capture and treats whitespace differently; the two must share one declared
  accepted-input and error contract before any accuracy comparison.
- **Live run with the new checker** (`stage4-runs/computer-3`, computer backend): `INTEGRATION CHECK PASSED`, exit 0;
  4 tool calls, all exact, each with its JSON-RPC id; final replies and token counts identical to computer-2; every
  request's raw reply saved; V100 peak 32,417 MiB again.
