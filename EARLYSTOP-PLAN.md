# Stopping failed long answers early (plan v1, 2026-11-01; for Chris's yes and Codex's review)

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
   answers have no per-token times, so on development data the saving is an estimate (each question's time scaled by
   the share of its text tokens after the stop); the fresh run measures it exactly. **Choose one rule and its parameters on this data, then freeze it** (code, parameters, sha256)
   before any fresh question exists.
2. **Fresh questions:** a new set from `stage5/generate.py` with a new seed (5003), the same groups and counts as the
   primary set (70/10/40/40), generated only after the rule is frozen. Within-group overlap with the 160 (the same
   templates, new numbers) is disclosed.
3. **The measurement, exactly and in one run:**
   - Greedy decoding is deterministic and a stop rule can only end an answer earlier, so the stopped answer is
     exactly a prefix of the unchanged one.
   - One run of **unchanged F** on the fresh set records every generated token's time (the runner's `on_token`, on
     the parent's clock).
   - The frozen rule is applied offline to each answer: the stop position, the prefix answer (graded by the frozen
     scorer, a stop counting as a failure) and the time to the stop.
   - A short live run with the rule switched on (a few fresh questions, chosen beforehand) checks that the live
     stopped answers equal the offline prefixes token for token.
4. **Registered comparison (on the fresh set):** F-stop vs unchanged F, paired.
   - **Accuracy:** a stop can only turn an answer F gets right into a failure, never the reverse (a stopped answer
     counts as wrong). So the accuracy cost is exactly the number of **premature stops**: each question is listed, and
     F-stop's score is F's minus that count. No interval is needed for it.
   - **Time:** the summed-time saving (the primary efficiency figure) and the median, both from the recorded token
     times.
   - **Registered verdict:** the rule is "worth keeping" if it saves at least 10% of F's summed time with at most 1
     premature stop in 160. Otherwise the result is reported as it is.
   - **Secondary:** the same rule applied offline to arms G, S and C's saved answers on the 160, descriptive only
     (development data).
5. **Validity:** as the fast run: frozen runtime and analysis identities, a preflight gate, lock errors invalidate the
   run, every attempt kept, no reruns.

## Stop rules
- If no candidate saves at least 5% of F's summed time on the development data without a premature stop there,
  stop and report: the gain isn't worth a fresh run.
- If the live check finds a live/offline difference, the offline evaluation isn't valid: report, no fresh-run claim.

## Not part of this
Changing the model, the prompts, the budgets of the unchanged arm, or anything on the computer side.
