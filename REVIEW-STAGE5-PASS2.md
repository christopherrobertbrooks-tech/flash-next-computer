# Pass 2 questions: stage 5 tooling

Only after pass 1 is pushed. Now read the `stage5` commit messages too.

1. **Where does this brief's account disagree with your pass 1?** Who's right?
2. **Code tests:** they run on Dev-Console, not the gateway, because the gateway's AppArmor
   (`kernel.apparmor_restrict_unprivileged_userns=1`) makes bwrap fail ("uid map" / "loopback"). On Dev-Console the
   wall unshares every namespace (no kernel-level network), binds `/usr` read-only and the test folder read-only,
   10 s timeout, 1 GB address space; sockets are also replaced inside Python. A deliberate escape attempt (network,
   writes to home and /usr, reading home) found nothing, and an endless loop was killed at 10 s. Enough for scoring
   model-written functions?
3. **Runner attribution:** a fake-server test showed that after a client timeout, the next question queued behind
   the old one and late tool calls (including a deliberate reference mismatch) landed on the wrong question. The fix
   waits on Strata's `GET /status` (busy/queued) before each question and after any failed one. Is `/status` the right
   signal in Strata 0.1.41 (`server.py` ~2624, ~2716)? Any remaining window?
4. **Classifier:** "call format" vs "backend" for a tool result that starts with "error" is decided by asking
   upstream's `calc.expected` on the gateway (ssh) whether the same expression is also an error there; if that can't be
   answered, it counts as call format. Fair? A refusal by the shared input contract is call format. The OOM case:
   HTTP error text containing "out of memory" is backend, but the 503 that follows an engine restart is transport.
   Should that be backend?
5. **Statistics:** Newcombe method 10 (uncorrected phi); self-test includes a 2,000-run coverage simulation (0.950).
   Is the 160-question primary set and the 5-point margin a sound pre-registration, given temperature 0 (one
   deterministic answer per question)?
6. **Chain evidence for arm A** comes from the reply and reasoning text (digits after removing commas and spaces), so
   it can over-count (a number written as part of the question restated). Report it as weak, or drop it?
7. **Group 2** ("near the computer's limits"): is it fair to keep it in the primary set, given it's designed to hit
   the computer's declared limits (so it favours B by design)? Or should it be reported separately and the primary
   set be groups 1, 3, 4 (150)?
