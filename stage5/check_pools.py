"""Checks every code-task expected output in generate.CODE_POOL against an independent reference solution."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from generate import CODE_POOL  # noqa: E402

REF = {
    "count_vowels": lambda s: sum(c in "aeiouAEIOU" for c in s),
    "second_largest": lambda nums: (sorted(set(nums))[-2] if len(set(nums)) > 1 else None),
    "digit_sum_factorial": lambda n: sum(map(int, str(math.factorial(n)))),
    "run_length": lambda s: "".join(f"{k}{len(list(g))}" for k, g in __import__("itertools").groupby(s)),
    "to_roman": lambda n: "".join(r * c for r, c in _roman(n)),
    "balanced": lambda s: _bal(s),
    "merge_intervals": lambda iv: _merge(iv),
    "nth_prime": lambda n: _prime(n),
    "flatten": lambda xs: _flat(xs),
    "binary_to_int": lambda b: int(b, 2),
    "transpose": lambda m: [list(r) for r in zip(*m)],
    "collatz_steps": lambda n: _collatz(n),
}


def _roman(n):
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, r in vals:
        out.append((r, n // v)); n %= v
    return out


def _bal(s):
    st, pair = [], {")": "(", "]": "[", "}": "{"}
    for c in s:
        if c in "([{": st.append(c)
        elif c in pair:
            if not st or st.pop() != pair[c]: return False
    return not st


def _merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out


def _prime(n):
    c, k = 0, 1
    while c < n:
        k += 1
        if all(k % d for d in range(2, int(k ** 0.5) + 1)): c += 1
    return k


def _flat(xs):
    return [y for x in xs for y in (_flat(x) if isinstance(x, list) else [x])]


def _collatz(n):
    s = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1; s += 1
    return s


bad = 0
for which, pool in CODE_POOL.items():
    for func, arg, what, args, outs in pool:
        for a, e in zip(args, outs):
            got = REF[func](eval(a[0]))   # noqa: S307  (our own fixed test inputs)
            ok = got == eval(e)           # noqa: S307
            bad += not ok
            if not ok:
                print("BAD", which, func, a[0], "expected", e, "reference", repr(got))
print("pool check:", "all expected outputs match the references" if not bad else f"{bad} mismatches")
sys.exit(1 if bad else 0)
