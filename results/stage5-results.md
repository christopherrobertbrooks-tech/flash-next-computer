# Stage 5 results: is the computer useful? (2026-10-30)

**Answer: no accuracy advantage over host Python was established** (within the registered +/-5-point tolerance).
On 160 maths questions, Flash-Next Coder with the exact computer got
156 right and with an ordinary Python calculator 155. The pre-registered verdict is **practically equivalent**: the
difference is +0.6 points, 95% interval -3.0 to +4.4, inside the agreed +/-5. The computer took 1.6x Python's
median time per question, inside the agreed median rule (at most 2x and 60 s), so it counts as practical. **Either
calculator roughly doubles what Flash-Next gets right on this maths set (alone: 81/160).** The computer never disagreed
with its reference machine, but 4 of its calls failed with CUDA out of memory (below).

Codex reproduced every score and the verdict independently (`review/codex-stage5-results`, ff80127 / 3d12646); this
page carries its wording corrections.

Protocol: `STAGE5-PLAN.md` v3, frozen at tag `stage5-freeze` (1d721dd) before the final questions existed; questions
`stage5/final-questions.jsonl` (seed 5002, sha256 `3f97781e`, generated after the tag). Run 10:46 am - 3:19 pm on the
V100 (one job in the lock; vision off; all 12,288 experts resident; temperature 0). Scored with `stage5/score.py`.
Raw data: `stage5-final/` (replies gzipped, every tool call, configs, memory, startup); scores:
`stage5-final/scores.json.gz`. Provenance limits (Codex): the tag, source equality, question hash, saved configs and
ordered logs are consistent and the deployed code matches the frozen hashes now, but there's no per-launch immutable
runtime snapshot, so unrecorded changes or attempts can't be excluded absolutely.

## Right answers
| Group | A alone | B Python | C computer |
|---|---|---|---|
| 1 big-number arithmetic | 22/70 | 68/70 | 68/70 |
| 2 near the computer's limits | 7/10 | 10/10 | 10/10 |
| 3 word problems | 35/40 | 40/40 | 40/40 |
| 4 dependent chains | 17/40 | 37/40 | 38/40 |
| **primary (1-4)** | **81/160** | **155/160** | **156/160** |
| 5a code with tests | 8/10 | 9/10 | 9/10 |
| 5b code reading | 10/10 | 9/10 | 10/10 |

**The primary comparison, C vs B (160 paired questions):** both right 153, computer only 3, Python only 2, both
wrong 2. Difference +0.0062, Newcombe 95% interval [-0.0298, +0.0441] -> **practically equivalent (within 5 points)**.
Without group 2 (descriptive only, 150 questions): 143 / 3 / 2 / 2, the same picture. All 5 disagreements are chain
questions where the two runs took different paths (below), not computer errors.

## Why answers failed (one layer each)
| Layer | A | B | C |
|---|---|---|---|
| budget / length termination (the 8,192-token cap per round, or Strata's reasoning-loop stop) | 75 | 3 | 3 |
| call format | 0 | 2 | 0 |
| final answer (tools fine, answer wrong) | 0 | 1 | 1 |
| reasoning (no tool, answer wrong) | 6 | 1 | 1 |

- **Alone:** of A's 81 failures, 75 ended as "length" without an accepted final answer and 6 stopped with a wrong
  answer. Length terminations include the token cap and Strata's reasoning-loop stop: the server log records 17
  loop stops in A, and 16 of its 75 length failures end below 8,192 tokens. An unfinished answer isn't evidence that
  it would have been right. B has one early loop stop (at 3,072 tokens).
- **Recovered tool faults** (a refused call or a computer limit, then the right answer): B 37, C 43. Mostly
  over-long calls refused by the shared 80-token input limit, which Flash-Next then split. In C, also the computer's
  "too complex: split it" twice (group 2), and once a set of 4 calls that failed with CUDA out of memory (below). A
  refusal or a "too complex" isn't a success of the tool; the later right answer is a success of the whole question
  under the frozen rule. The OOMs came back as tool errors inside successful HTTP replies, which v3 treats as
  recoverable; an HTTP-level OOM would have been a hard failure.
- **No reference mismatches** (0 in C), no harness errors, no transport errors, no fence stops, and no repeated-call
  loops by the registered rule (that rule counts identical tool calls; it says nothing about loops in the reasoning).
- **Call accounting, C:** 534 tool-call attempts (warm-ups included) = 487 complete reference-clean results, 43
  refusals, 4 execution errors (the OOMs).
- **Normal work (group 5) didn't get worse with a tool:** A 18/20, B 18/20, C 19/20 (tripwire: none). Tools were
  called on group 5 only once (C, `final-5b-007`, the out-of-memory one).

## Time and memory
| | A | B | C |
|---|---|---|---|
| median s per primary question | 58.2 | 11.6 | 18.9 |
| mean s per primary question | 51.2 | 18.0 | 25.0 |
| generated tokens per primary question (mean) | 5,514 | 1,762 | 1,724 |
| tool calls per primary question (mean) | 0 | 3.3 | 3.3 |
| total time in the tool | - | 0.2 s | 1,160 s |
| whole arm (182 requests) | 2.3 h | 0.83 h | 1.15 h |
| cold start (first warm-up) | 12.3 s | 10.4 s | 12.7 s |
| V100 peak (sampled every 500 ms) | 32,005 MiB | 32,011 MiB | 32,467 MiB |
| tool process peak host RAM | - | 1,386 MiB | 1,664 MiB |

- **"Practical" holds as a median rule:** C's median is 1.63x B's and under 60 s. It's not a per-question cap:
  primary requests over 60 s number A 79, B 6, C 9 (C's longest 101.9 s). The computer ran at 1,686 circuit
  tokens/s (including its reference check); its ~1,160 s of tool time over the arm is the main cost.
- **Memory is tight.** The sampled device maximum was 32,467 MiB of 32,768 in C; that isn't a safe margin: on
  `final-5b-007` four calls failed (`CUDA out of memory ... 28 MiB free`), down to `2+37`, and Flash-Next answered right
  without them. Reserving room (`--vram-reserve-mib 1200`) worked on stage 4's prompts at a cost of ~255 expert slots;
  it hasn't been measured on the full stage-5 workload, and it isn't shown to remedy the divergence below.

## Repeatability, and an open question
- **Python run twice:** the registered 40-question repeat of B matched B on **40/40** final replies. That supports
  repeatability on that subset, not every output token or all model state.
- **B and C differ more than the tool results explain.** Final replies were identical on 141 of 182 requests. On 21
  of the 41 that differ (20 chains, one code-reading question), the first tool call already differs, with the reasoning
  diverging within its first 8-1,440 characters: before any tool result existed. The requests, first prompt token
  counts, the single prefill chunk, zero reuse and startup free memory (660 MiB) were the same in both. Those match
  counts aren't a saved token-ID or state-byte equality. The tool list rendered into the prompt is the same in both by
  source (name, description and schema; the backend's server name isn't rendered), but input token ids weren't saved.
- **Where:** groups 1 and 3 match B completely (final and reasoning text); group 2 differs once (the legitimate "too
  complex" case). All 40 chain reasoning texts differ (27 final texts). Later, many tool-free code and code-reading
  questions differ too (5a: 6/10 final texts match; 5b: 1/10). The first differing chain is `final-4-001`, scored
  question 121 (request 123 counting the two warm-ups). That is also where the task family changes, so the boundary
  isn't only elapsed session time.
- **Diagnostic repeat of C** (Chris's go; post-hoc, not part of the registered protocol; `stage5-final/diag-Crep/`,
  5:18-5:36 pm, the same 40 questions as B's repeat): it matched **Python on 40/40 final texts** (all 40 right). It
  differs from the original C run on **8/40 final texts** (10 counting reasoning): `final-4-001`, `-005`, `-009`,
  `-013`, `-021`, `-029`, `-033`, `-037`. Only `final-4-029` changes correctness: wrong (length) in the original C,
  right in the diagnostic, so the original-C subset is 39/40 vs 40/40. The diagnostic doesn't replace any primary
  result.
- **What that does and doesn't show:** the original long C run diverged on later questions; a fresh, short C run
  matched Python on its 40-question subset. That rules out "the computer process always changes the output". It
  **doesn't** rule out an effect that depends on the run's length, history or shared-GPU resources, and it doesn't
  identify a mechanism. **The cause remains unresolved.** No engine restart or diagnostic explaining the drift was
  found in the logs; tool errors, including the four OOMs, were recorded.
- Candidates and the evidence that would separate them (Codex): the computer's own embedding cache, which grows
  across calls (per-call allocated/reserved memory beside a device timeline); Strata's K/V and checkpoint layout
  (checkpoint ids, positions, layouts at the first differing question); expert-slot refill contents and readiness
  (residency checksums); MTP/suffix/verify paths or CUDA concurrency (draft sources, window shapes, graph state at the
  first differing token); prompt rendering (saved input token ids and tool-schema hashes). None was measured here.

## Chains
Both tool arms got almost every chain right (B 37/40, C 38/40), but Flash-Next often batches several steps into
one expression or does the last step in its reply, so "steps seen in order in the tool results" averages 33% (B) and
28% (C). That measures how it uses the tool, not whether it was right.

## What this answers in Chris's plan
The finish line: "Flash-Next Coder on the V100 able to call an exact computer, with a measured answer to whether it
does better than Flash-Next alone and Flash-Next with host Python."
- **Able to call it:** yes (stage 4; here 534 call attempts over 182 requests: 487 complete reference-clean results,
  43 refusals, 4 out-of-memory errors; no reference mismatches).
- **Better than alone:** yes, by a lot (156 vs 81 of 160), but so is Python (155).
- **Better than host Python:** no accuracy advantage was established within the +/-5-point tolerance (the interval's
  upper end still allows a small difference, and this is one firmware and task design). Practically equivalent in
  accuracy, 1.6x slower by median, ~450 MiB more VRAM, and one question with four out-of-memory call failures. Its only extra is that every result is checked against a reference machine. That's a
  property, not an accuracy gain, and this design couldn't test the plan's "exact state across many steps" idea
  (the calc firmware has no memory between calls).
