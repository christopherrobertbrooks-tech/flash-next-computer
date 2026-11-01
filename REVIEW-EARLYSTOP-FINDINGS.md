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

## Pass 2 — guided (after pass 1 pushed as 8caae02)

Read REVIEW-EARLYSTOP-PASS2.md, results/earlystop-dev.md and the remaining plan only after the blind commit was
successfully pushed. No experiment run was started.

### Guided answers

1. **The development choice is acceptable and disclosed.** The post-grid choice criterion is stated explicitly:
   maximize estimated F saving subject to zero premature stops on all four development arms. Checking every raw
   candidate confirms exactly two qualify, (64,6) and (64,8); the former saves 376.1 vs 375.1 estimated seconds on F.
   My blind preference for (64,8) was a different conservative tie-break, not a defect in their stated development
   choice. This is tuning on development data, not a pre-registered discovery or four independent confirmations.
   The 11.84% estimate rounds correctly to 11.8%, and the warning about being close to 10% is appropriate.

   Replaying the actual frozen rule over the rebuilt own-token streams matches earlystop_dev.py's stop positions
   on all 640 saved replies. The five F failure stop fractions are 6.73%, 8.04%, 12.14%, 30.03%, 44.85%, consistent
   with the page. All three named misses are correct, but their explanation needs the small correction below.

2. **The adjusted live categories are sound.** A pure repetition rule cannot exercise a new lower-budget boundary.
   Selecting untriggered normal-budget failures tests its non-trigger behavior, alongside two actual triggers and
   median/long EOS completions. The procedure was frozen before fresh outcomes. Shortfalls are disclosed; no
   unregistered substitute category is needed. The actual fresh file is in lexicographic ID order, so taking the
   first two in run order implements the stated ID order for this set. These checks do not replace run validity.

3. **Fresh numbers in the same templates are enough for the stated narrow test.** Confirmed zero identical prompts
   and zero identical answer strings against inside4b/final-primary.jsonl, with 160 unique new prompts and answers.
   Regeneration matches the saved seed-5003 records, and all chain finals equal their final intermediates. This
   evaluates the frozen rule on another sample from the same generator; it does not test unseen problem families
   or supply a population accuracy guarantee. The plan's disclosure and operational verdict are appropriately
   scoped.

4. **Hold the experiment sequence pending pass-1 redlines 1–3.** The stop-rule choice, question freshness and rule
   freeze are not blockers. Fix validity, timing validation and the final-verdict gate before spending the fresh
   run; add the corresponding failing controls and refresh the affected analysis/runtime identities. Keep the
   selected rule and fresh question file unchanged. Updating infrastructure identities before any outcomes exist
   does not require retuning the rule or generating another question set. A passing live check should authorize
   an *estimated* saving claim; it does not make that estimate an exact, repeatable live speedup or guarantee a
   10% realized reduction on future workloads. Do not run earlystop.sh as part of this review.

### Additional findings

4. **NIT — the misses have too few repetitions, not necessarily none.**
   Draft: development page calls the three misses “long answers without an exact 64-token repeat”.
   Evidence: among rebuilt own-token spans, final-1-060 has maximum multiplicity 1, final-2-003 has 2, and
   final-4-039 has 4. The rule needs 6. Replace with: “long answers without six occurrences of the same 64-token
   span in the rebuilt model-owned token stream.” This does not change the catches or the gate.

5. **NIT — qualify the token-count closeness as aggregate and round upward.**
   Draft: “the rebuilt counts are within 0.6% of the recorded ones”.
   Evidence: F/G/S/C aggregate deficits are 0.528%/0.505%/0.533%/0.647%; individual discrepancies are larger
   (e.g. F final-3-015 loses 163 of 8,192 recorded text tokens, about 1.99%). Replace with:
   “The rebuilt aggregate counts are within 0.7% of the recorded totals; individual replies can differ more.”
   Aggregate agreement is not evidence of exact token-ID reconstruction.

### Additional CPU checks

After pass 1, ran the six supplied reference-machine hook controls (section A) read-only on the gateway's existing
runtime, with CUDA_VISIBLE_DEVICES empty and PYTHONDONTWRITEBYTECODE=1. Extracted only the non-writing test body:
no scratch directory, no model load, no earlystop.sh. Both modes pass recording invariance, exact ownership/index
checks (41 decisions in each), and stopping at the expected prefix without another helper call. This strengthens
the hook validation beyond pass 1's minimal fakes; the full scratch-writing A/B/C/D script was not run there.
The local C/D controls and negative reproductions remain as recorded in pass 1. bash -n earlystop.sh also passes.

Final review verdict: **changes requested before the unchanged/scoring/live sequence proceeds**. Three REDLINEs
remain; no GPU execution is needed to fix or verify them. Only this findings file changed on the review branch.

## Confirm — c0305f0

Merged origin/earlystop (c0305f07fb972d22c09f9e0f659f7f505be4c54e) into review/codex-earlystop as requested.
Followed REVIEW-EARLYSTOP-CONFIRM.md: counterexamples/controls first, then inspected the fix diffs and identities.
Review only, CPU only. No earlystop.sh, model load, GPU access, or gateway writes.

### Findings 1–5

1. **Closed — live-run validity.** The shared run_problems gate is used before scoring and for both runs at live
   validation. It checks completion/shutdown status, F/token mode, decision recording, baseline/live early-stop
   settings, registered budgets, verified artifact count and the exact runtime-manifest digest. Lock, trace and
   harness errors invalidate the run. Live question-file SHA and selected IDs are checked. Re-ran matching-token
   counterexamples with incomplete startup, wrong arm/mode, zero verified files, another manifest, wrong budget,
   wrong early-stop setting, lock error and harness error: each is rejected and the final verdict is withheld.

2. **Closed — final-verdict gate.** Offline evaluation emits candidate_verdict and explicitly withholds the final
   verdict. The analysis live path persists final.json with validity, per-question results and the final decision.
   A valid matching live run can promote the candidate; every tested validation failure with a candidate keep
   produces a withheld final verdict. No live run means no final keep verdict.

3. **Closed — elapsed-time and decision validation.** The decision timestamps and total duration now use
   time.monotonic. Evaluation rejects negative, NaN, decreasing and past-duration times, zero duration, mismatched
   token IDs, repeated/out-of-order indices and an out-of-range index on an EOS answer. The permitted terminal
   unappended decision is explicitly handled. The original all-minus-100-second dataset is rejected. Negative live
   duration and invalid live decision times are rejected. Ran the actual run_questions function body on scripted
   CPU fakes with a backward wall-clock jump: its legacy wall seconds became -200, while its new monotonic duration
   and decision times remained valid; evaluation uses the latter, so that wall-clock jump cannot create savings.

4. **Closed — explanation of misses.** The page now correctly says the missed answers lack six occurrences of the
   same rebuilt model-owned 64-token span, with multiplicities 1, 2 and 4.

5. **Closed — reconstruction precision.** The page now states aggregate counts are within 0.7%, discloses larger
   individual discrepancies (about 2%), and says reconstruction is not exact token-ID replay.

### Verification

- 19 supplied evaluator/live controls (C/D) passed locally, extracting that body to avoid GPU/runtime imports and
  keeping scratch in local /tmp. Added 11 independent checks of the earlier counterexamples and verdict behavior:
  30 checks passed in total. The actual-runner backward-clock check above also passed.
- Mutation test: temporarily replaced run_problems with a no-op in the local test process. The supplied negative
  control for incomplete/shutdown-failed live runs failed immediately, detecting an incorrect final keep. Restored
  the function; no repository source was edited. This confirms those controls can detect a disabled validity gate.
- All 11 analysis identity hashes verify at frozen commit 9e5d5dcf660a. Repository runtime hashes match the updated
  manifest (including the changed runner). Both v1-superseded manifests are byte-for-byte copies of the old versions.
- Rule hash remains c29310c9ee4106d42fb93f7c47bb17748dc2ca41958f05d11b4ae8277322ab6f; fresh-question hash remains
  c66d4729526fd57feea555d6055e1d2b204ed80e034a8f5bdfc0490c8b5866fc. Neither file changed since the original freeze.
- bash -n earlystop.sh and git diff --check pass. The hook implementation did not change, so its previously passing
  reference-machine controls were not repeated. No GPU measurement or full gateway runtime rehash was performed;
  the launcher's required artifact check/preflight still runs before the experiment.

**Confirmation verdict: findings 1–5 closed; no new findings. earlystop.sh unchanged may proceed through its
registered lock, artifact and preflight gates.** Then score the unchanged run and perform the registered live check;
only a valid passing check can release the final verdict. This review did not start any part of that sequence.
