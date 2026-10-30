# Stage 5 pilot (2026-10-30, 10:20-10:42 am)

15 pilot questions (seed 5001, sha256 `eee173f5`, never part of the final set) in all three arms, each its own Strata
launch in the V100 lock, the reviewed tooling (`stage5/`, Codex-cleared at 644c54b). Raw data: `stage5-pilot/`;
scores: `stage5-pilot/scores.json` (`stage5/score.py`).

## It ran cleanly
- No stops: no computer-vs-reference mismatch, no harness errors, no fence waits, no transport errors.
- Code tests ran in the wall on real model output (group 5a: 2/2 in every arm).
- Peak V100: A 32,003, B 32,009, C 32,457 MiB (all 12,288 experts resident; the computer adds ~450 MiB, ~310 MiB
  headroom). Tool process peak RSS: Python 1,394 MiB, computer 1,654 MiB.
- Cold start (first warm-up): A 12.3 s, B 10.4 s, C 12.6 s. Strata ready 30 s after launch, the tool 38 s.

## Results (12 primary questions: far too few for the verdict; this checks the tooling and timing)
| Group | A alone | B Python | C computer |
|---|---|---|---|
| 1 arithmetic | 1/3 | 3/3 | 3/3 |
| 2 near limits | 2/3 | 3/3 | 3/3 |
| 3 word problems | 3/3 | 3/3 | 3/3 |
| 4 chains | 1/3 | 3/3 | 3/3 |
| 5a code | 2/2 | 2/2 | 2/2 |
| 5b code reading | 1/1 | 0/1 | 0/1 |
| **primary (1-4)** | **7/12** | **12/12** | **12/12** |
| median s / primary question | 50.7 | 15.7 | 44.3 |

- **B vs C:** 12 both right. The scorer correctly says "inconclusive" (+/-24 points at n = 12). Replies and tool
  calls were identical on 16 of 17 requests. On the 17th (`pilot-2-002`), the computer said "too complex: split it",
  Flash-Next split the sum into 4 calls, and wrote the same final text.
- **Recovered faults** (right answers after a tool fault): B 2, C 3. Two over-capture refusals in each, and the
  computer's "too complex" once.
- **A's failures** are all the 8,192-token budget: 5 of 12 primary questions ran out while still thinking.
- **5b (code reading):** A answered right in 23 s; with a tool offered (B and C) the same question thought past 8,192
  tokens. One question: noted, not a pattern yet.
- **Chains:** both tool arms got all three right, with every step called except, usually, the last (done in the
  reply), so "all steps seen in order" is 0/3 by its strict definition. Descriptive only, as planned.
- **Time:** C's median is 2.8x B's (44.3 vs 15.7 s). By the pre-registered rule (at most 2x, at most 60 s), the
  computer is "impractical" on this sample. Its tool time was 162 s over the pilot vs 0.012 s for Python.

## Projected full run
Per-question medians A ~50 s, B ~16 s, C ~44 s -> 180 questions x 3 arms + B's 40-question repeat + startup:
**about 5.5 hours**, inside the 3.5-7 hour estimate.
