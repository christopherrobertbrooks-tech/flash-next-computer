# Stage 5 results: is the computer useful? (2026-10-30)

**Answer: no measurable gain over Python.** On 160 maths questions, Flash-Next Coder with the exact computer got
156 right and with an ordinary Python calculator 155. The pre-registered verdict is **practically equivalent**: the
difference is +0.6 points, 95% interval -3.0 to +4.4, inside the agreed +/-5. The computer took 1.6x Python's
median time per question, inside the agreed 2x and 60 s, so it's practical. **Either calculator roughly doubles what
Flash-Next gets right on big-number maths (alone: 81/160).** The computer never disagreed with its reference machine.

Protocol: `STAGE5-PLAN.md` v3, frozen at tag `stage5-freeze` (1d721dd) before the final questions existed; questions
`stage5/final-questions.jsonl` (seed 5002, sha256 `3f97781e`, generated after the tag). Run 10:46 am - 3:19 pm on the
V100 (one job in the lock; vision off; all 12,288 experts resident; temperature 0). Scored with `stage5/score.py`.
Raw data: `stage5-final/` (replies gzipped, every tool call, configs, memory, startup); scores:
`stage5-final/scores.json.gz`.

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
| budget (ran out of the 8,192-token room, per round) | 75 | 3 | 3 |
| call format | 0 | 2 | 0 |
| final answer (tools fine, answer wrong) | 0 | 1 | 1 |
| reasoning (no tool, answer wrong) | 6 | 1 | 1 |

- **Alone, Flash-Next almost never gets the maths wrong; it runs out of room.** 75 of its 81 failures were the
  token budget (doing 20-30 digit multiplications by hand in its thinking); only 6 were wrong answers.
- **Recovered tool faults** (a refused call or a computer limit, then the right answer): B 37, C 43. Mostly
  over-long calls refused by the shared 80-token input limit, which Flash-Next then split. In C, also the computer's
  "too complex: split it" twice (group 2), and once a set of 4 calls that failed with CUDA out of memory (below).
- **No reference mismatches** (0 in C), no harness errors, no transport errors, no fence stops, no loops.
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

- **"Practical" holds:** C's median is 1.63x B's and under 60 s. The computer ran at 1,686 circuit tokens/s
  (including its reference check); its ~1,160 s of tool time over the arm is the whole cost.
- **Memory is tight.** With all experts resident, the computer leaves ~300 MiB on the V100. On `final-5b-007` it ran
  out (`CUDA out of memory ... 28 MiB free`) and 4 calls failed; Flash-Next answered right without them. A safer
  setup reserves room for it (stage 4: `--vram-reserve-mib 1200`, at a cost of ~255 expert slots).

## Repeatability, and an open question
- **Python run twice is identical:** the registered 40-question repeat of B matched B on **40/40** replies. Strata
  is deterministic from one launch to the next in this setup.
- **But B and C differ more than tools can explain.** Final replies were identical on 141 of 182 requests. On 21
  of the 41 that differ, the runs had already split **before any tool result existed**: the first tool call differs,
  with the reasoning text diverging within its first 8-1,440 characters. The prompt and its reading were identical in
  both (same token count, same single prefill chunk, nothing reused), and startup memory was identical (660 MiB free).
  Six of these questions are in the repeat subset, where B matched itself exactly. So **the computer's presence on the
  card seemed to change Flash-Next's output somewhere in Strata's generation**, even though no tool result had been
  delivered. This isn't a tool-correctness problem: C's answers were still right on 156/160. Twenty of the 21 are chain
  questions (long prompts, long thinking); the pilot and stage 4 (short) showed it once or never. The diagnostic
  repeat below narrows it down.
- **Diagnostic repeat of C** (Chris's go; not part of the registered protocol; `stage5-final/diag-Crep/`, 5:18-5:36 pm,
  the same 40 questions as B's repeat): the computer arm on its own matched **Python on 40/40** replies, and differed
  from the first C run on 9, exactly the 9 where that first run had split from Python. So **the computer's presence
  doesn't change Flash-Next's output**; something in the long first C run did.
- **Where it happened:** in the first C run, the first 121 requests (groups 1-3) match Python, apart from the one where
  the computer legitimately said "too complex". From request 122 (the first chain, 2:35 pm) to the end, nearly every
  reply differs, including the last 20 coding questions, where no tool was called at all. So Strata switched into a
  different state at about that point and stayed there. Python's long run (and both short repeats) never did.
- **Not found in the logs:** no warnings, restarts or errors in either server log; the same expert-slot lending and
  refilling; the same resident-RAM lines; V100 memory (sampled every 500 ms) in the same ranges in the long and the
  short computer runs. The cause is still open. Something accumulates over a long session when the computer shares the
  card; the out-of-memory moment at the end of that run suggests memory headroom, but that isn't shown. The accuracy
  result stands either way (the divergence added 5 discordant pairs, split 3-2, inside the interval). A practical
  safeguard is to reserve room for the computer (`--vram-reserve-mib 1200`).

## Chains
Both tool arms got almost every chain right (B 37/40, C 38/40), but Flash-Next often batches several steps into
one expression or does the last step in its reply, so "steps seen in order in the tool results" averages 33% (B) and
28% (C). That measures how it uses the tool, not whether it was right.

## What this answers in Chris's plan
The finish line: "Flash-Next Coder on the V100 able to call an exact computer, with a measured answer to whether it
does better than Flash-Next alone and Flash-Next with host Python."
- **Able to call it:** yes (stage 4; here 534 tool calls over 182 requests; every call the computer ran was
  checked against its reference machine, with no mismatches).
- **Better than alone:** yes, by a lot (156 vs 81 of 160), but so is Python (155).
- **Better than host Python:** no. Practically equivalent in accuracy, 1.6x slower, ~450 MiB more VRAM, and one
  out-of-memory failure. Its only extra is that every result is checked against a reference machine. That's a
  property, not an accuracy gain, and this design couldn't test the plan's "exact state across many steps" idea
  (the calc firmware has no memory between calls).
