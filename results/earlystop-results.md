# Stopping failed long answers early: results (2026-11-01, 3:53-5:12 am)

Builds on [Qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal (MIT); please cite it as given in
the [README](../README.md#credits-and-citation).

**Answer:** the frozen repetition rule (stop when the language model's own text repeats the same 64-token span for
the 6th time) is **kept** under the registered criterion. On 160 fresh questions, applied to unchanged arm F (the
exact computer with the stock Qwen3.5-4B Q8_0), it **stopped 7 of the 8 answers that ran to the text budget** and
**no answer F got right** (0 premature stops; 131/160 either way). It **saved an estimated 12.2% of the summed time**
(4,013 to 3,522 s). The live check passed, and with it the final verdict. Reviewed by Codex (review/codex-earlystop bfb39b6 /
a53ee69): every number, the replay and the live check reproduce; the KEEP verdict stands; one wording correction
applied below.

Protocol: `EARLYSTOP-PLAN.md` v2.
- **The rule:** `llamacpp/earlystop_rule.py`, sha256 `c29310c9...`. It was chosen on development data
  (`results/earlystop-dev.md`, rough development saving 11.8%) and frozen before the fresh questions existed.
- **The questions:** `results/earlystop/fresh-questions.jsonl` (seed 5003, 70/10/40/40, sha256 `c66d4729...`; no
  prompt or answer equal to the 160 registered ones; the same templates with new numbers).
- **Frozen before the runs:** the runtime identity (112 files), matched at both startups, and the analysis identity
  (11 files), verified before scoring and before the live check.
- **Preflight:** the gate passed (99.687%).
- **Run status:** ok, with 0 lock errors.

| | unchanged F | F with the rule (estimated) |
|---|---|---|
| right answers | 131/160 | **131/160** (0 premature stops) |
| summed time | 4,013 s | **3,522 s** (12.2% less) |
| median time | 8.27 s | 8.02 s |
| text-budget failures | 8 | 1 left (7 stopped) |

**What the rule stopped:** all 7 of these were budget failures that unchanged F gets wrong. The times are F's
unchanged question time, then the estimated time with the rule.

| Question | Unchanged | With the rule | Saved | Stopped after token |
|---|---|---|---|---|
| fresh-1-031 | 193.8 s | 45.2 s | 148.6 s | 42,313 |
| fresh-4-031 | 93.0 s | 7.9 s | 85.2 s | 665 |
| fresh-4-040 | 93.1 s | 20.1 s | 73.0 s | 1,754 |
| fresh-4-019 | 96.0 s | 28.7 s | 67.3 s | 7,491 |
| fresh-4-023 | 93.1 s | 29.1 s | 64.0 s | 2,564 |
| fresh-4-009 | 149.0 s | 120.7 s | 28.3 s | 85,690 |
| fresh-4-035 | 119.7 s | 94.8 s | 24.9 s | 52,303 |

(Token counts include the computer's chip tokens.) The one budget failure it doesn't stop is fresh-4-007.

## The live check
The fixed procedure picked:
- 2 questions the rule stops (fresh-1-031, fresh-4-009);
- 1 that reaches the 8,192 budget without a rule stop (fresh-4-007; the only one, a shortfall of 1);
- 2 that end at EOS: fresh-3-024 (median length) and fresh-2-004 (the longest).

They ran with the rule switched on. Both runs passed the validity checks.

| Question | Tokens vs offline | Early stop live / offline | Live time vs estimate (limit) |
|---|---|---|---|
| fresh-1-031 | equal (42,313) | yes / yes | 46.41 vs 45.20 s (49.72) |
| fresh-4-009 | equal (85,690) | yes / yes | 120.67 vs 120.66 s (132.73) |
| fresh-4-007 | equal (55,003) | no / no | 120.27 vs 120.25 s (132.28) |
| fresh-3-024 | equal (3,598) | no / no | 6.45 vs 6.44 s (8.44) |
| fresh-2-004 | equal (83,063) | no / no | 125.18 vs 125.46 s (138.01) |

The live stopped answers are token-for-token the offline prefixes. Across the five checks, live durations differed
from the offline estimates by -0.28 to +1.22 s, and all passed the registered tolerance. Those differences include
rule overhead and run-to-run variation; they don't isolate overhead. `results/earlystop/eval/final.json`.

## What it means
- **The rule works as designed on this test set.** It ends most of the runaway answers, often very early (fresh-4-031
  after 665 tokens), and on these 160 questions it never cut an answer F got right.
- **Limits:**
  - One run, one question sample from the same generator, so it isn't a guarantee for other question families or
    future answers.
  - The saving is estimated from recorded stop points; the live check confirms the boundary and passes the
    registered timing tolerance on five questions. A full stopped-run saving wasn't measured.
  - The keep criterion (at least 10% saved, at most 1 premature stop) is an operational decision on this set.
- **Stopped answers are failures:** the rule doesn't fix those answers; it stops spending time on them. The eight
  budget failures took 957.9 s (23.9% of the time); the rule removes about half of that (51.3%). The median moves
  only about 3% (8.27 to 8.02 s): the saving is in the failed tail.
- **Adoption:** we adopt this frozen rule as the operational default for F: on one fresh 160-question sample from the
  same generator, it met the pre-set keep criterion with 12.2% estimated summed-time savings and no observed
  premature stops. This doesn't guarantee zero losses on future answers. It stays switchable (`--early-stop`), the
  unchanged baseline stays runnable, and it applies to arm F only. Making it the runner's default is a separate,
  tested change.
