# Stage 3 interface proof: the computer as a service (2026-10-30)

`computer/computer.py`: Qwenputer's RV32 circuit on upstream's Triton executor (fp32 circuit arithmetic), driven with
no language model: a call feeds the tool-call text as the model would write it, the circuit runs the calc-bridge
firmware and writes the response, and **every circuit token is checked against upstream's reference machine**
(`bridge.Session`); the call stops at the first difference. Built only from the checkpoint's config, chipformer.json
and tokenizer (no weights). `computer/test_computer.py` (log `computer-service-test.log`, raw
`computer-service-results.json`): V100, lock clocks.

| Group | Calls | Reference-clean (every token) | Equal to Python (upstream `calc.expected`) |
|---|---|---|---|
| Random, upstream's generator: add, sub, mul, div, mod, gcd, pow, expr, frac, rat (seed 1) | 70 | **70** | **70** |
| Malformed / edge | 11 | **11** | 10 |
| Five calls in one session | 5 | **5** | **5** |
| Long (62-digit) operands | 2 | **2** | 1 |
| Budget stop at 50 computer tokens | 1 | stopped as designed | n/a |

The two non-matching answers aren't computer errors (the computer matched its reference machine on every token):
`"  42  "` gives 42 from the firmware but "error" from upstream's Python helper (the two references treat surrounding
spaces differently); a 62-by-62-digit product is "error (too long: split it)" from the firmware and "error" from Python
(both refuse: the result exceeds the firmware's 62-digit limit).

**Speed:** 319,722 computer tokens over the clean calls at **1,707 tokens/s**; median call **1.56 s**, longest 8.9 s
(a 14,068-token rational expression). For comparison, Python computes these in microseconds: the computer's value has
to come from somewhere else than speed (stage 5 measures whether it does).

**Stage 3 gate (plan):** zero trace mismatches on the declared tests: yes; correct call/result routing including
malformed input, sequences and a budget stop: yes; memory and idle behaviour: measured (434 MiB as its own process;
idle it costs Strata ~11% on the test request; with vision off that equals today's vision-on speed).
