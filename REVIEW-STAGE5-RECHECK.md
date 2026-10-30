# Re-check brief for Codex: stage 5 tooling after your findings (fa30ce4 / 97e0f86)

Same rules as before: your own clone, branch `review/codex-stage5` (rebase or merge `stage5` into it), review only,
no GPU, no Strata, nothing pushed to `main`, `stage3` or `stage5`. This is a guided re-check, not a new blind pass.

What changed, finding by finding (details in `STAGE5-PLAN.md`, "v3 changes"):
1. Code tests (`stage5/score.py`, `run_tests`): the child gets only the code and the call strings; it returns plain
   exact-type values; the parent checks exit status, the result shape and compares with the expected values. A wall
   canary runs before scoring; a bwrap failure is a harness error.
2. `newcombe_paired`: the covariance correction; fixtures (10,5,2,20), (80,0,0,80), (160,0,0,0); coverage at 2%, 10%
   and 30% disagreement (2% over-covers at 0.997: conservative, only a floor is checked).
3. Fencing (`stage5/ask.py`, `computer/mcp_server.py`): calls tagged with their question at arrival (begin and end
   lines); fence = Strata idle + every begun call ended (+ 10 s quiet after a failure); fail closed (exit 5). Every
   attempt saved. `stage5/test_ask_fake.py` is the CPU control set (a late call ending after `/status` says idle
   included).
4. Typed errors (`computer/computer.py`, `mcp_server.py`): refused / budget / trace_mismatch / reference_error /
   execution_error; only trace_mismatch stops the run. Backend exceptions are logged too.
5. Failure rule (`score_one` and the plan): hard failures first, whatever the final text; recovered faults counted
   separately; loops flagged.
6. Chains: steps seen in order (one result per step) and in any order; none for arm A.
7. `stage5/generate.py` checks group 1 against `computer/contract.py` with the real tokenizer (runs on the gateway
   CPU); group 2 labels `over_capture`.
8. `load_arm` / `main`: duplicates, wrong arm, unknown ids, another question file rejected; verdict withheld if
   anything is missing; the B repeat must be exactly the registered 40.
9. The Python reference for an "error" result is recorded by the tool server at call time; unknown stays "unknown".
10. Rounds and generated tokens from the full engine log per question; tool pid and peak RSS logged per call; the
    `int(b, 2)` ban dropped.

Also decided (your pass-2 answers): group 2 stays primary with a descriptive groups-1/3/4 breakdown; "practically
equivalent" wording; an explicit OOM is backend, a later 503 stays transport.

Please check: do the fixes close 1-10, do the new controls actually fail when they should, and is anything still in
the way of a pilot? Append "Re-check" to `REVIEW-STAGE5-FINDINGS.md` and push.

## Second re-check (after 993affb): findings 11-15
- 11: `run_tests` no longer reads stderr (discarded); a nonzero exit runs the trusted canary, and only a failing
  canary raises `WallError`. Control: candidate writes "bwrap: ..." to stderr and exits 1 -> a wrong answer.
- 12: `mcp_server.py` writes an `instance` id (pid + start time) on every begin and end line; `ask.py`'s
  `read_calls` / `open_calls` / row collection match on (instance, rpc_id), flag duplicate keys, ends without begins
  and lines without an instance; the runner also fences after an OK reply if one of its calls is still open, and
  records open calls when it stops. Controls in `test_ask_fake.py`: a restart reusing an RPC id whose call ends after
  /status says idle; a call that never ends (exit 5, open call recorded).
- 13: `load_arm` requires a readable `questions.sha256` matching the question file (absent / empty rejected).
- 14: `strict_equal` compares types recursively (tuples-as-lists the only allowance). Controls: bool for 1/0 fails,
  real ints pass, float for int fails.
- 15: output read in chunks with a 1 MB cap (100 MB control stopped by the cap); `python_reference` returns None on a
  helper exception (scored "unknown").
Real tool server smoke on the gateway CPU (Python backend): begin/end lines carry the instance and question tag.
Please re-check 11-15 only and append to `REVIEW-STAGE5-FINDINGS.md`.
