# Early stop: the development analysis and the frozen rule (2026-11-01)

Development data only: the saved answers of the 160 registered questions for arms F, G, S and C
(`llamacpp/earlystop_dev.py`, raw grid `results/earlystop-dev.json.gz`). The saved replies have text, not token ids,
so tokens were rebuilt by splitting each reply into the circuit's forced frames and the model's own text and
re-tokenizing; the rebuilt counts are within 0.6% of the recorded ones (F 202,024 vs 203,096). Time savings are
**rough**: each stopped question's time scaled by the share of its text after the stop.

## The grid (premature stops = answers the arm got right that the rule stops)
| Rule | F premature | F failures stopped | F saved | G / S / C premature |
|---|---|---|---|---|
| budget 6,144 | 1 | 8/8 | 6.8% | 0 / 0 / 0 |
| budget 5,120 | 1 | 8/8 | 10.4% | 0 / 1 / 2 |
| budget 4,096 | 2 | 8/8 | 14.2% | 2 / 4 / 5 |
| repetition 32 tokens x 6 | 0 | 5/8 | 12.9% | 0 / 2 / 2 |
| repetition 32 x 8 | 0 | 5/8 | 11.9% | 0 / 0 / 1 |
| repetition 64 x 4 | 0 | 6/8 | 12.8% | 0 / 0 / 3 |
| **repetition 64 x 6** | **0** | **5/8** | **11.8%** | **0 / 0 / 0** |
| repetition 64 x 8 | 0 | 5/8 | 11.8% | 0 / 0 / 0 |
| budget 6,144 + repetition 64 x 6 | 1 | 8/8 | 14.9% | 0 / 0 / 0 |

(All 64 combinations are in the raw grid. Every rule with a lower budget cuts at least one of F's right answers;
shorter spans or fewer repeats cut many.)

## The choice
- **The criterion,** set when the grid was read, on development data: the largest F saving with **zero premature
  stops on all four arms**. Only repetition 64 x 6 and 64 x 8 qualify; 64 x 6 saves slightly more.
- **The plan's development gate** (at least 5% of F's time saved without a premature stop) passes.
- **On F's eight failures,** it stops final-4-032 at 7% of its length, final-4-001 at 8%, final-3-027 at 12%,
  final-4-007 at 30% and final-3-015 at 45%. It doesn't stop final-1-060, final-2-003 or final-4-039 (16 calls):
  long answers without an exact 64-token repeat.
- The 10% keep threshold is close to this rough 11.8%, so the fresh run may land either side of it.

## Frozen
- `llamacpp/earlystop_rule.py`: `RepetitionStop(k=64, r=6)` over the model's own tokens, consulted before each
  Qwen-chosen token is appended, EOS never refused; sha256 `c29310c9ee4106d42fb93f7c47bb17748dc2ca41958f05d11b4ae8277322ab6f`.
- On all 640 saved answers its stop points equal the development script's.
- The fresh questions are generated after this freeze.
