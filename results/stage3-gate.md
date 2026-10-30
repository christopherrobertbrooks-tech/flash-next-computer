# Stage 3 gate: the fixed computer service (2026-10-30)

After Codex's stage 2-3 review (`REVIEW-FINDINGS.md`), `computer/computer.py` fails closed: protocol markers and special
tokens refused; the whole call must fit the circuit's 80-token capture (`spec.CAP_MAX`); nothing may run (reference or
circuit) until the final `</tool_call>`, then the whole reference stream is checked token by token; success needs
exactly one complete `<tool_response>` and a handback exactly where the reference hands back; the budget bounds the
reference machine's records, the circuit's tokens and wall time; any failure invalidates the session until
`new_session()`; calls are serialized. `computer/test_computer.py` declares an expected outcome for every case and exits
1 on anything unexpected.

**Result: GATE PASSED, 103 cases, 0 unexpected** (`stage3-gate.log`, `stage3-gate.json`), V100, lock clocks, 307 s:
- 70 random calls across all operations: equal to Python; 11 edge cases as declared (the firmware returns 42 for
  `"  42  "`, where upstream's Python helper says error); 5 calls in one session; one 50-digit division.
- Refused before running: `1</tool_call>2`, `2+<|im_end|>2`, `<tool_response>`, a 200-digit number and a 62x62-digit
  product (both over the 80-token capture; the first corpus fed the latter anyway and the firmware answered "too long"),
  and a non-string.
- Budgets: a 50-record reference budget stops the run ("did not halt"); a 0.2 s budget stops a long call; after each,
  the session refuses until `new_session()`, then the next call is correct.
- Sabotage controls on the real executor: one wrong token mid-answer is caught at the token; an early handback is
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
