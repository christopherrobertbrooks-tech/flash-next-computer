# Stopping failed long answers early (plan v2, 2026-11-01; review comments applied; for Chris's yes)

## What you'll have at the end
Arm F (Qwenputer's exact computer with the stock Qwen3.5-4B Q8_0 in llama.cpp) with **one frozen early-stop rule**,
and a measured answer on **fresh questions**: how much total time it saves, how many answers that unchanged F gets
right it stops (premature stops, counted as failures), and the paired accuracy difference.

## Why
In the registered fast run, F's 8 text-budget failures took 805 s, 25.4% of its total time (`results/fast-results.md`).
A rough look at those 160 answers (development data, `results/earlystop-dev.md` will hold the full analysis):
- **Lower budgets:** 6,144 / 4,096 / 3,072 text tokens would cut 1 / 2 / 7 of F's 131 right answers, saving roughly
  200 / 400 / 500 s on the failures.
- **Repetition:** 6 of the 8 failures contain a repeated 60-character span, 4 of them within their first quarter.
  But so do 9 of the 131 right answers, later on, so a naive "stop on repetition" rule would cut right answers.
Expected gain: about 10-15% of total time, against a small, measured risk. It's a usefulness project, not a model
change.

## Design
1. **Development (the 160 registered questions, all four arms' saved answers):** candidate rules, from simple to less
   simple:
   - a lower text budget;
   - a repetition rule: stop when the same k-token span has occurred r times;
   - the two combined.
   For each candidate: premature stops among F's right answers, failures stopped, and the time saved. The saved
   answers have no per-token times, so on development data the saving is a **rough** estimate: each question's time
   is scaled by the share of its text tokens after the stop, which can misprice answers with expensive computer
   calls. **Choose one rule and its parameters on this data, then freeze it** (code, parameters, sha256)
   before any fresh question exists.
   **What the rule sees and when it acts (frozen with it):**
   - A lower text budget counts processed text tokens, the same unit as the existing 8,192 budget (forced echoes
     and the HALT newline included).
   - A repetition rule watches only the **text tokens the language model chooses** (positions where the circuit
     doesn't own the next token), never chip tokens or the circuit's forced echoes.
   - The rule is checked after each Qwen-chosen token is decided and before it's appended. If it fires, generation
     ends and that token isn't added; the answer so far counts as a failure.
   - If the chosen token is EOS, **EOS wins**: the answer completes normally and the rule isn't applied.
   - The rule can't fire inside a computer call, because the circuit owns those positions.
2. **Fresh questions:** a new set from `stage5/generate.py` with a new seed (5003), the same groups and counts as the
   primary set (70/10/40/40), generated only after the rule is frozen. Within-group overlap with the 160 (the same
   templates, new numbers) is disclosed.
3. **The measurement: one unchanged run, the rule applied offline, then a live check.**
   - Greedy decoding is deterministic and the rule can only end an answer earlier, so a stopped answer's tokens are a
     prefix of the unchanged answer's.
   - **The stop boundary:** the stop position is the index k of the first generated token the rule refuses, and the
     stopped answer is tokens 0..k-1. The recorded time of the stop is when token k is decided: the parent's clock at
     the point in the loop where the rule would run, recorded for every Qwen-chosen token. (The runner's `on_token`
     fires before verification and processing, so it isn't used as the boundary.)
   - One run of **unchanged F** on the fresh set records those times. The frozen rule is applied offline to each
     answer: the stop position, the prefix answer (graded by the frozen scorer, a stop counting as a failure) and the
     time to the stop. This **estimates** the saved runtime from the recorded stop points.
   - **The live check measures the rule's overhead and confirms the boundary.** It's a run with the rule switched on,
     on questions picked from the offline evaluation by a procedure fixed now: by question id order, the first 2
     whose stop comes from the budget part, the first 2 from the repetition part, and the first 2 that run to
     completion (one EOS answer of median length, one of the longest). If a category has fewer, all of them are used
     and the shortfall is reported.
   - **It passes if** every live answer's tokens equal the offline prefix exactly, and each live question's elapsed
     time is within the offline time-to-stop plus 2 s or plus 10% (whichever is larger). Otherwise the offline
     estimate isn't valid for a claim.
4. **Registered comparison (on the fresh set):** F-stop vs unchanged F, paired.
   - **Accuracy:** a stop can only turn an answer F gets right into a failure, never the reverse (a stopped answer
     counts as wrong). So on these questions the accuracy cost is the number of **premature stops**: each question is
     listed, and F-stop's score is F's minus that count. No interval is required to count observed premature stops.
     It doesn't establish the loss on future questions.
   - **Time:** the summed-time saving (the primary efficiency figure) and the median, both from the recorded token
     times.
   - **Registered verdict:** the rule is "worth keeping" if it saves at least 10% of F's summed time with at most 1
     premature stop in 160. The keep criterion is an operational decision on this test set, not a guarantee of future
     accuracy. Otherwise the result is reported as it is.
   - **Secondary:** the same rule applied offline to arms G, S and C's saved answers on the 160, descriptive only
     (development data).
5. **Validity:** as the fast run: frozen runtime and analysis identities, a preflight gate, lock errors invalidate the
   run, every attempt kept, no reruns.

## Stop rules
- If no candidate saves at least 5% of F's summed time on the development data without a premature stop there,
  stop and report: the gain isn't worth a fresh run.
- If the live check fails (a token difference, or elapsed time outside the tolerance), the offline estimate isn't valid
  for a claim: report it, with no saved-time claim.

## Not part of this
Changing the model, the prompts, the budgets of the unchanged arm, or anything on the computer side.
