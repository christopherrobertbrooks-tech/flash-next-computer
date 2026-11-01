# Early-stop review

## Pass 1 — blind (03c2006)

Scope: REVIEW-EARLYSTOP.md, the plan only before “The frozen rule”, the named code/diffs, raw development grid,
frozen manifests and fresh questions. Did not open the development write-up, pass-2 brief, or later commit messages.
Review only; no earlystop.sh, GPU job, model load, or gateway writes.

### Development choice and freeze

The 64-entry grid reproduces exactly using earlystop_dev.py and the compatible base tokenizer, on CPU. Five rules
have zero premature stops on F and exceed the 5% estimated development gate:

| k | r | F budget failures stopped | Estimated seconds saved | Estimated saving |
|---|---|---|---|---|
| 32 | 6 | 5/8 | 409.0 | 12.88% |
| 32 | 8 | 5/8 | 377.0 | 11.87% |
| 64 | 4 | 6/8 | 406.3 | 12.79% |
| 64 | 6 | 5/8 | 376.1 | 11.84% |
| 64 | 8 | 5/8 | 375.1 | 11.81% |

No pure budget or combined candidate has zero F premature stops. The plan does not prescribe an optimization or
tie-breaking objective, so there is no unique required winner. From the grid alone I would favor (64,8): nearly the
same estimated gain as (64,6), requiring more repetition. This is a conservative preference, not evidence about
future losses. The implemented (64,6) is also an admissible development choice; no finding for choosing it.

Rebuilding tokens is adequate for choosing a hypothesis, not exact trajectory verification. F rebuilds 202,024 vs
203,096 recorded text tokens; per-question discrepancies range from -163 to +1. The other arms also differ. All 640
replies have the expected number of forced-response frames relative to recorded tool calls. The approximation and
text-proportional time estimate are disclosed, and the fresh run uses actual decision IDs. Do not present the dev
zero-loss figure as an exact replay of the original model IDs.

Freeze chronology is supported by independent artifacts: rule hash
c29310c9ee4106d42fb93f7c47bb17748dc2ca41958f05d11b4ae8277322ab6f is in b077679
(03:15:25 -0600); the gateway development snapshot has that hash and mtime 03:15:04; fresh questions have mtime
03:19:28, after the generator change at 03:19:20. The rule is unchanged in 03c2006. Fresh questions enter Git in
15948c5, before the analysis manifest in 02c610f. These support the recorded order, not a claim that filesystem or
Git timestamps can prove nobody generated an unrecorded set earlier.

All 11 analysis-manifest hashes verify. Runtime-manifest hashes for run_split.py, split_chat.py, earlystop_rule.py,
earlystop.sh and fresh questions match repository files. The manifests cover the core model/helper/executor/scorer
inputs; accepting a run as using those identities still needs the checks below. The 160 fresh records have unique
IDs/prompts and counts 70/10/40/40; regenerating seed 5003 with compatible base token IDs and the capture contract
reproduces every record exactly.

### Findings

1. **REDLINE — the live check can pass an invalid run.**
   Spec: “lock errors invalidate the run” and “frozen runtime and analysis identities” (plan Design 5).
   Draft: earlystop_live_check.py:15–39 reads tokens, flags and seconds, but no startup validity; analysis.py's `live`
   path never calls `validity()` on either run. A synthetic matching live run still passes after adding an
   `incomplete` startup (helper shutdown failed, wrong arm/mode, zero verified artifacts) and lock/harness errors
   to all reply rows. The existing token/flag/time negative controls do not cover this failure.
   Smallest fix: validate both run directories before reporting LIVE CHECK PASSED, requiring expected arm/mode,
   baseline/live early-stop settings, normal completion/shutdown, and no lock/trace/harness errors. Bind startup to
   the expected runtime manifest by digest, not merely its file count; check the registered budgets and selected
   question identity. Add controls for invalid startup, final shutdown failure, trace error and wrong runtime.

2. **REDLINE — “keep” is emitted before the required live validation.**
   Spec: “Otherwise the offline estimate isn't valid for a claim” (plan Design 3).
   Draft: earlystop_eval.py:59 immediately sets `verdict: keep` from saving and premature stops; analysis.py `score`
   writes and prints it before a live run exists. The later `live` command neither withholds nor updates that
   verdict if validation fails. Thus the reporting artifact can retain a successful verdict after a failed check.
   Smallest fix: label the offline result `candidate_keep` (or provisional), with the final verdict withheld pending
   a successful, valid live check tied to that exact evaluation. Persist final validation/verdict together. A failed
   live check must produce a withheld final verdict even when the offline thresholds pass. Test both outcomes.

3. **REDLINE — impossible decision times are accepted as savings.**
   Spec: the recorded stop time is “when token k is decided”, on the question's elapsed clock (plan Design 3).
   Draft: earlystop_eval.py:38–59 accepts every recorded time without finite/range/order checks; run_split.py uses
   adjustable wall time. On the supplied four-question evaluator fixture, changing decision times to -100 produces
   estimated time -178 s, saving 217.11%, and `verdict: keep`. The live check also accepts negative live elapsed
   times when tokens match because it checks only an upper limit.
   Smallest fix: use a monotonic elapsed clock and fail closed on nonfinite/negative or decreasing timestamps,
   timestamps beyond the question duration (allowing its explicit rounding tolerance), nonpositive baseline totals,
   and invalid live durations. Validate decision indices/token correspondence too, explicitly accounting for the
   existing terminal unappended decision at text-budget exhaustion. Add negative-time, NaN, reversed-time and
   inconsistent-index controls before trusting savings.

### Checks and limits

Ran 12 local CPU controls/reproductions: the six supplied evaluator/live controls (C/D, extracted without the
GPU-dependent imports), invalid-live and impossible-time reproductions, actual split_generate body on minimal
scripted fakes for recording/prefix behavior, EOS precedence in the rule, and the terminal decision at text-budget
exhaustion. The actual generator recording hook preserved tokens/counts; refusing the frozen-rule decision yielded
exactly the offline prefix. Code inspection confirms circuit-owned choices bypass the hook, and no hook is supplied
when both options are absent. EOS exemption is in the actual rule.

Did not run the full supplied A/B reference-machine controls: the local interpreter lacks PyTorch/upstream runtime.
Did not substitute a GPU run for that limitation. Gateway access was read-only tokenizer/hash/timestamp reads under
the house-rules skill. These are build findings before the fresh experiment; there are no fresh outcomes to score.
