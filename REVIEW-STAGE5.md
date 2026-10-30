# Review brief for Codex: stage 5 tooling, before any GPU time (two passes)

Stage 5 asks whether Flash-Next Coder gets more right with the exact computer (arm C) than with a Python calculator
(arm B) or alone (arm A). `STAGE5-PLAN.md` is the protocol (v2, after Muse's and your review of v1). This review
covers the tools that will run it; nothing has run on a GPU yet.

## Where to work
Your own clone, branch `review/codex-stage5` from `stage5`. Not `~/Code/flash-next-computer`. On the gateway,
read-only: Strata `/mnt/steam/Strata-0141` (`serve/server.py`, `serve/mcp.py`), upstream Qwenputer
`/mnt/nvme/qwenputer/qwenputer/experimental/riscv` (`chipformer/rv/calc.py`, `env/calc/calc.c`). Don't start
Strata or use a GPU; don't change files there.

## Pass 1: blind (commit and push before opening `REVIEW-STAGE5-PASS2.md`)
**Read:** `PLAN.md` (stage 5 section), `STAGE5-PLAN.md`, `stage5/` (generate.py, check_pools.py, ask.py,
run_arm.sh, score.py), and the code they reuse: `computer/stage4_requests.py` (ANSWER parser),
`computer/mcp_server.py`, `computer/contract.py`.
**Don't read yet:** commit messages on `stage5`, or `REVIEW-STAGE5-PASS2.md`.

Run what needs no GPU:
- `python3 stage5/generate.py pilot X.jsonl` and `final Y.jsonl` (the final set is only for review here; it isn't
  frozen yet and won't be committed until after the pilot);
- `python3 stage5/check_pools.py`;
- `python3 stage5/score.py --selftest`;
- `computer/stage4_requests.py --selftest`.

Then answer:
1. **Does the tooling implement the protocol?** Arms, prompts, budgets, warm-up, subset, stop rules, failure layers,
   the Newcombe interval and the verdict rule: where does code differ from `STAGE5-PLAN.md`?
2. **The questions:** are the answers right? Do any questions leak their answer, or fall outside both tools' limits
   when they shouldn't? Are the groups what they claim (e.g. do chains really need each previous exact result)?
3. **The scorer:** can a wrong answer pass, or a right one fail? Is each failure put in the right layer? Is the
   chain "reached" measure sound? Is the code-test wall adequate?
4. **The runner:** can tool calls, timeouts or engine errors be attributed to the wrong question? Can a run silently
   lose data?
5. Anything that would make the final numbers misleading.

Write `REVIEW-STAGE5-FINDINGS.md` ("Pass 1 (blind)"), commit and push.

## Pass 2: guided
Open `REVIEW-STAGE5-PASS2.md`, answer it as "Pass 2 (guided)" in the same file, push.

## Ground rules
Review only: no GPU, no Strata launches, no downloads; don't push to `main`, `stage3` or `stage5`.
