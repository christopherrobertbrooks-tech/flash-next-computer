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

# Stage 4 robustness (2026-10-30, about 6:10-6:30 am)

All runs use the computer backend and the enforcing checker (`computer/stage4_requests.py`). The same four prompts
were sent each time, and the outputs were compared with `computer-3`. Raw files are in `stage4-runs/r-*`.

| Run | What changed | Checks | Outputs vs computer-3 | V100 peak |
|---|---|---|---|---|
| r-two-passes | the prompt set twice in one server session | 8/8 met | **identical** in both passes (text, token counts, reasoning length) | 32,419 MiB |
| r-no-mtp | `--mtp` removed (suffix drafts only) | 4/4 met | **identical** | 31,253 MiB |
| r-no-reuse | `--prompt-cache 0` (every round re-reads the whole prompt) | 4/4 met | different (see below) | 32,427 MiB |
| r-vision | normal vision config + the computer | **2/4: engine out of memory** | n/a | 32,447 MiB |
| r-vision-reserve | vision config, `--vram-reserve-mib 1200` (was 700) | 4/4 met | different (fewer resident experts) | 32,129 MiB |

- **Two conversations in a row:** the second pass resumed from the first pass's checkpoints. For example, the
  multiplication took 4.5 s instead of 10.8 s, and every output was identical. Resuming from saved state
  didn't change any answer.
- **MTP on/off:** identical outputs, as Strata's design says (speculative verification is bit-identical to greedy).
- **`--prompt-cache 0` is not a clean pause/resume test.** It also changes how Strata reads a *fresh* prompt: with it
  off, Strata keeps no turn-boundary checkpoint ("0 checkpoints" vs "1"). The haiku and the first round of the
  `(2**64-1)//7` prompt then come out differently, although nothing was reused in either run (both "0 reused").
  The difference therefore comes from how the prompt is read, not from resuming after a tool round. A direct state
  comparison (GDN/KV after a pause vs a cold read) still needs engine access and stays **open**.
- **Vision + computer: doesn't fit as configured.** With vision, Strata leaves 486 MiB free; the computer takes
  ~430 MiB, then Strata builds its verify graphs on the first answer (48, then 28 MiB free) and fails ("verify: instantiate: out
  of memory"). The engine exits, restarts sized around the computer (11,438 slots), and the next two prompts pass.
  **Fix:** reserve room up front. With `--vram-reserve-mib 1200`, the run had 11,387 slots (normal vision: 11,642;
  -255, ~2%), 984 MiB free after load, no errors, and all checks met. Fewer resident experts send some expert work to
  the CPU, so outputs differ from the all-resident runs. The haiku also completes in both runs that changed the read
  path; neither difference is a tool effect.
- **Cancellation:** not supported or claimed (the MCP server handles one request at a time and ignores cancel
  notifications).
- **Frozen inputs:** `inputs-manifest.txt` (`scripts/inputs_manifest.sh`) holds sha256 of the firmware, calc source,
  bridge/executor code, Qwenputer config/tokenizer files, the Strata engine binaries, server/MCP code, both configs, the
  expert profile and our service/driver code. The 13 multi-GB weight files get size, mtime and a sha256 of their first
  and last 64 MiB ("partial"). Also recorded: upstream qwenputer `ca13310` (one local change: the low-RAM patch in
  `chat.py`), Strata `fb58e0d`, torch 2.14.1 / triton 3.8.0 / transformers 5.22.0, driver 580.173.02.

## One contract for both backends (before stage 5)
`computer/contract.py` now holds the accepted-input rules, and **both** MCP backends use them: protocol markers,
special tokens and calls over the 80-token capture are refused alike. The Python backend also reads input the way the
firmware does (newlines dropped, surrounding spaces skipped). `computer/contract_test.py` runs the same inputs through
both backends exactly as the MCP server calls them: 70 random calls (seed 2, not the gate's) plus 56 tricky ones.

- First comparison (`contract/contract-1.log`): 118/126 the same. 3 whitespace differences were Python's helper
  choking on leading spaces and newlines; those were fixed in the Python backend.
- Second (`contract-2.log`): **121/126 the same; the 5 differences are all declared, as the computer's real limits**
  (firmware `calc.c`), exit 0:
  - two long random expressions: "error (too complex: split it)": the firmware runs out of registers;
  - `10**61`: the firmware's `**` multiplies through intermediate products that must fit 62 digits (Python checks only
    the final value);
  - `1\t+2` (a tab) and `1_000` (digit separators): Python-only syntax.

  Stage 5 must count these: on such inputs, the computer answers "error" where Python gives a value.
- The gate after moving the rules into `contract.py`: 118/0, exit 0 (`stage3-gate3.log`).

## Codex review of 60a4a33 (c2d2b77) and the fixes
- **19 (redline): the answer check accepted wrong answers.** It searched for the digits anywhere in the reply, so
  `-121932631112635269`, `121932631112635269.5` and "Not 121932631112635269; the answer is 0" all passed. Fixed:
  the prompts now ask for a final `ANSWER: <the exact value>` line. The checker parses the last such line as one
  integer, decimal or p/q (after removing markdown, commas and LaTeX `\frac`) and compares it exactly. Tool correctness
  and final-answer correctness are now separate verdicts. `--selftest`: 25 cases, including those three wrong answers,
  a missing ANSWER line and a non-numeric one, all classified correctly; a deliberately broken copy exits 1. Runs made
  before the ANSWER line can only be re-checked with a weaker "legacy" check (the expected number as a whole signed
  token somewhere in the reply), reported as "mentioned (weak)", not final-answer correctness. All saved runs pass it,
  except r-vision's two out-of-memory requests.
- **Live, with the ANSWER line** (`stage4-runs/a-computer`, `a-python`): both `INTEGRATION CHECK PASSED`, tool and
  answer verdicts met on all three arithmetic prompts. The computer and Python replies are again identical (text and
  token counts) on all four prompts. The new prompts changed the replies themselves (e.g. the two-step now makes 3
  calls), so these two runs are the new baseline.
- **20: the inventory missed Flash-Next's tokenizer and chat template.** Fixed: every regular file under the pack is
  included, recursively (vocab, merges, token types, `chat_template.jinja`, tokenizer metadata, which match Codex's
  independent fingerprints). Files over 256 MiB are labelled partial fingerprints (first/last 64 MiB), which can't
  detect a change in the middle: this is provenance, not a full-content freeze of the weights.
- **21: declared differences were broader than observed.** Fixed: `contract_test.py` now accepts only the five exact
  observed (expression, computer outcome, Python outcome) triples. `--selftest` proves that `2+2` -> "too complex",
  `2**2` -> error, a tab case where Python crashed, a different `10**61` result, and a `1_000` computer failure are
  all undeclared. Re-applied to the saved contract-2 data: still 121/126 same, 5 declared.
