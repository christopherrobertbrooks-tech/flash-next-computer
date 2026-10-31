# Re-check pass 2 (open only after pass 1 is pushed)

What I changed, and the claims to check:

Viewer (`viewer-live`):
1. **Committed snapshot (finding 1):** `Shadow` keeps a private checker machine `cm`; the screen reads `Shadow.m`, a
   `Snap` copy (x, ram, bn, out, anchor/n, steps, halted, fault, exit_code, parked; shared rom), replaced only in
   `_commit()` after a record and all its echoes matched. A mismatch leaves it at the last verified state.
   Control: an independent `Witness` (its own bridge Session, capturing call text, stepped one record per reported
   commit) must equal the display at every token; a subclass restoring the old design (display = checker) is caught
   on all 1,297 mid-record tokens of the fixture.
2. **Termination (finding 2):** `Shadow.finish(reason)`; the worker reports `reason` (eos / text budget / computer
   budget / stopped / error, plus the error text). Only `eos` with nothing unfinished (`in_call`, `pending`,
   `expect`) is "VERIFIED, answer complete"; everything else is INCOMPLETE; mismatch stays UNVERIFIED. Headless exits
   1 unless complete. EOF fixtures: after `</tool_call>`, after GO, inside a multi-token record, BYTE before its echo,
   HALT before its newline; whole stream ended by each reason. Is the eos/budget classification in
   `start_model.generate` right for upstream's `generate` return values?
3. **Cancellation (finding 3):** `make_producer` (timed puts that check the stop flag, the done item too),
   `stop_producer` (sets stop, drains, joins, reports whether the thread ended), called in a `finally` around the
   curses UI. Credit reset when toggling run/pause; credit is spent only while running. Docstring now says pause
   pauses playback, with backpressure at 4,096 tokens. Controls: a full 8-slot queue stopped without a consumer, a
   producer error, a normal end.
4. **Labels (finding 9):** source "SHADOW vs fp32 executor" + "bf16 layers NOT run; calc-bridge.elf" on two short
   lines; status counts "stream tokens checked (N computer)"; BMAC with ovw shows `=`; replay label states the
   capture approximation; "n next call" only in replay; a parked machine shows "parked (waits for the next call)".
5. **Premise wording (finding 8)** in `VIEWER-PLAN.md`.
6. Live check after the fixes: `results/viewer/live-check-2026-10-31.txt` (2**64 // 7, eos, 4,641 chip tokens,
   VERIFIED complete).

Speed (`computer-speed`): `results/speed-results.md` and `SPEED-PLAN.md` rewritten with your pass-2 wording (your
summary paragraph verbatim; synthetic readback loop; bookkeeping not isolated; 9.84% apply from your saved profile,
my unsaved 21% withdrawn; fusion ordering caveat; draft/verify not disproven plus the layer-major idea; unmatched
text table; the Amdahl range 28.2 / 34.2 / 42.2%; fla cold vs warm, bf16 cause not shown, broken detector). Did I
carry your corrections faithfully, and is anything still overstated? The "85% of C's question time" and "~12.4%"
figures are 6,438.5 / 7,620.8 and 943.65 / 7,620.8.

Also: today's three commits were first pushed with the wrong author email and re-pushed (force-with-lease) as
ba358d5 / c9114e7 with the noreply identity; content unchanged.

Append "Re-check pass 2" to `REVIEW-VIEWER-SPEED-FINDINGS.md`, commit, push, and say which redlines are closed.
