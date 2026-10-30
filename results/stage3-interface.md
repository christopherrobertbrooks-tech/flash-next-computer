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

**Speed:** the 88 clean calls produced 319,671 computer tokens in 187.222 s (**~1,707 tokens/s**), median **1.552 s**,
longest 8.935 s (`4/(-203)*962-40*31-(-37)*(-4)`, 15,325 tokens); with the budget-stop call, 319,722 tokens. For comparison, Python computes these in microseconds: the computer's value has
to come from somewhere else than speed (stage 5 measures whether it does).

**Not yet the stage 3 gate** (Codex's review): this first corpus proof had fail-open paths in the service (reference
output during input feeding could be dropped, the budget wasn't end-to-end, failed sessions stayed usable) and in the
runner (it didn't fail on failures), and its memory figures came from the stand-in holder, not this service. See
`stage3-gate.md` for the fixed service and the enforced gate.
