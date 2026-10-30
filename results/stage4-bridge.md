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
- **The computer leaves no trace beyond its answer.** With the computer and with Python, Flash-Next produced the same
  reply text and the same token counts for all four prompts. Python run twice was identical too, so Strata is
  deterministic here, and the comparison means something.
- **Prefix reuse works after each tool round.** For example, "prompt 557 tokens = 526 reused" after the first call:
  Strata did not re-read the conversation.
- **Cost: time.** Each computer call took 1.4-3.2 s; Python took about 0.1 ms. The first computer request was also
  about 5 s slower than the call itself accounts for. Most likely it waited for the computer to finish loading (Strata
  starts the MCP server in the background after it says "ready"); this is not yet measured separately.
- **Memory:** the computer process adds about 430 MiB. With all 12,288 experts resident, the V100 peaked at
  32,417 of 32,768 MiB: it fits, with about 350 MiB to spare. The script tried to fix the expert cache at 12,000
  slots, but `--resident-experts` overrides `--expert-cache` (the log still says 12,288 slots). The override was
  removed and the script now says so.
- The haiku prompt ran out of tokens while thinking (2,560 tokens, all thinking; `finish: length`, empty answer). It does
  this with both backends and never calls a tool, so this is Flash-Next's thinking, not the bridge.

## What this does and doesn't show
It shows the bridge works end to end: Flash-Next on the V100 calls an exact computer, gets checked answers, and is
otherwise unchanged. It does **not** yet show usefulness: on these prompts, the computer and Python give identical
outcomes, and Python is over 10,000x faster per call. Stage 5 measures whether anything differs on a real task set
(Flash-Next alone vs with Python vs with the computer).

Still open from Codex's stage 4 list: pause/resume isolation, cancellation, two consecutive conversations, MTP on/off,
end-to-end latency on more prompts, vision + service memory, and a frozen inputs manifest.
