# Stage 3 gate: the fixed computer service (2026-10-30)

After Codex's stage 2-3 review (`REVIEW-FINDINGS.md`), `computer/computer.py` fails closed: protocol markers and special
tokens refused; the whole call must fit the circuit's 80-token capture (`spec.CAP_MAX`); nothing may run (reference or
circuit) until the final `</tool_call>`, then the whole reference stream is checked token by token; success needs
exactly one complete `<tool_response>` and a handback exactly where the reference hands back; record and token caps
are enforced; calls are serialized. (The first version's claims about wall time and "any failure" were too broad:
see the re-check section below.) `computer/test_computer.py` declares an expected outcome for every case and exits
1 on anything unexpected.

**Result: GATE PASSED, 103 cases, 0 unexpected** (`stage3-gate.log`, `stage3-gate.json`), V100, lock clocks, 307 s:
- 70 random calls across all operations: equal to Python; 11 edge cases as declared (the firmware returns 42 for
  `"  42  "`, where upstream's Python helper says error); 5 calls in one session; one 50-digit division.
- Refused before running: `1</tool_call>2`, `2+<|im_end|>2`, `<tool_response>`, a 200-digit number and a 62x62-digit
  product (both over the 80-token capture; the first corpus fed the latter anyway and the firmware answered "too long"),
  and a non-string.
- Budgets: a 50-record reference budget stops the run ("did not halt"); a 0.2 s budget stops a long call; after each,
  the session refuses until `new_session()`, then the next call is correct.
- Sabotage controls on the real executor: the first emitted token wrong is caught at that token (`at: 0`); an early handback is
  caught; both recover after `new_session()`.
- 100 calls in one session: all correct.
- **The gate bites:** the same run with one deliberately wrong expectation (`2+2` -> "5") exits 1
  (`stage3-gate-injected.log`).

**The service's own memory** (not the stand-in holder): GPU process 398 MiB built, 448-450 MiB after the corpus, idle
and after 100 calls (torch allocated ~50 MiB, reserved 62-64 MiB); host RSS 1,686 MiB (peak 1,684-1,686). No growth
while idle or across the 100 calls. The reference history grows ~1,770 tokens per call (176,780 per 100 calls); the
2,000,000-token session cap means a new session every ~1,100 calls.

**Speed:** 91 clean calls, 322,879 computer tokens, 190.2 s (~1,697 tokens/s, including the reference check).

Still open before calling stage 3 fully closed (Codex): exact bridge routing through a real Flash-Next call, a vision +
service measurement, and pause/resume isolation (those belong to the stage 4 proof).

## Re-check of `a7ac5b3` (Codex, 352b583) and the fixes

Codex found two redlines left and two nits:
1. **An unexpected exception or a failed reset left the session marked valid.** Fixed: input refusals still change
   nothing, but past them the session is marked invalid *before* any state changes and becomes valid again only after
   a fully successful call or reset.
2. **The time limit skipped the input and the reference run.** Fixed: `max_seconds` is now one deadline
   (`time.monotonic`) from the moment `call()` is entered. It is checked while waiting for the computer (a busy
   computer refuses without touching the session), before each input token, before every reference-machine record, and
   before each circuit step. It is cooperative: one record or one step can overrun it, not more.
3. **The benchmark scripts exited 0 on failures, and Ctrl-C didn't stop them.** Fixed: HTTP, JSON, schema and
   readiness failures set a nonzero exit; INT/TERM exit (cleanup runs on EXIT); the holder log is truncated so an old
   "ready" line can't count. Probe: the reply checker on `{"error":"server failed"}` now exits 1 (it exited 0).
4. **Wording:** the claims above were corrected.

**Result: GATE PASSED, 118 cases, 0 unexpected, exit 0** (`stage3-gate2.log`, `stage3-gate2.json`), 314 s. The same
run with a deliberately wrong expectation: 119/1, `GATE FAILED`, **exit 1** (`stage3-gate2-injected.log`). New
positive controls, all caught as declared:
- the executor raises an exception mid-call -> the call raises, the next call is refused ("no valid session"),
  `new_session()` recovers;
- `new_session()` fails part-way (after changing everything) -> the next call is refused, a good reset recovers;
- a slowed reference machine (10 ms per record) with a 0.5 s limit -> stopped "in the reference run" in under 1 s;
- a busy computer (lock held) with a 0.3 s limit -> refused; the session stays usable (the next call gives 4);
- an input refusal leaves the session usable (`5*5` -> 25 right after a refused call);
- a wrong token deep in the answer (the 500th emitted token) is caught at that token (`at: 499`), alongside the
  first-token and early-handback controls.

Memory unchanged: GPU process 398 MiB built, 448 after the corpus and idle, 450 after 100 calls; host RSS 1,405-1,547
MiB (`ru_maxrss` peak 1,685). 97 clean calls, 331,474 computer tokens, 195.0 s.
