"""Stage 5 questions (STAGE5-PLAN.md). Pure Python, seeded, no model involved.

usage: python generate.py pilot OUT.jsonl     (15 questions, seed 5001, its own task pools)
       python generate.py final OUT.jsonl     (180 questions, seed 5002; generate only after the freeze)

Each line: {"id", "group", "prompt", "answer"} plus, by group: "intermediates" (chains), "tests" + "func" (code).
Answers are exact: an integer, a reduced fraction "p/q", or a decimal without trailing zeros. Every answer and every
intermediate stays within 62 digits (the calc tool's limit, in both backends), and every group-1 product within 62.
"""
import json, random, sys
from decimal import Decimal, localcontext
from fractions import Fraction

FINAL = " End your reply with one line: ANSWER: <the exact value>."
MAXD = 62
COUNTS = {"pilot": {"1": 3, "2": 3, "3": 3, "4": 3, "5a": 2, "5b": 1},
          "final": {"1": 70, "2": 10, "3": 40, "4": 40, "5a": 10, "5b": 10}}
SEEDS = {"pilot": 5001, "final": 5002}


def num(rng, lo, hi):
    """A random integer with lo..hi digits (no leading zero)."""
    d = rng.randint(lo, hi)
    return rng.randint(10 ** (d - 1), 10 ** d - 1)


def fmt(v):
    if isinstance(v, Decimal):
        s = format(v, "f")
        return s.rstrip("0").rstrip(".") if "." in s else s
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def fits(*vals):
    """Every value within the calc tool's 62 digits (numerator and denominator; a decimal's digit count)."""
    for v in vals:
        if isinstance(v, Decimal):
            if len(fmt(v).replace("-", "").replace(".", "").lstrip("0")) > MAXD:
                return False
            continue
        v = Fraction(v)
        if len(str(abs(v.numerator))) > MAXD or len(str(v.denominator)) > MAXD:
            return False
    return True


def dec(rng, lo, hi):
    """A decimal with lo..hi integer digits and 1..6 decimals."""
    return Decimal(f"{num(rng, lo, hi)}.{rng.randint(1, 999999):06d}".rstrip("0") or "0")


# ---- group 1: exact arithmetic (within both tools' limits) ------------------------------------------------------
def g1(rng):
    while True:
        kind = rng.choice(["mul", "mul", "add", "sub", "floordiv", "mod", "pow", "gcd", "frac", "dec"])
        if kind == "mul":
            a, b = num(rng, 10, 30), num(rng, 10, 30)
            text, v = f"{a} * {b}", a * b
        elif kind == "add":
            a, b = num(rng, 20, 40), num(rng, 20, 40)
            text, v = f"{a} + {b}", a + b
        elif kind == "sub":
            a, b = num(rng, 20, 40), num(rng, 20, 40)
            text, v = f"{a} - {b}", a - b
        elif kind == "floordiv":
            a, b = num(rng, 25, 40), num(rng, 5, 15)
            text, v = f"{a} // {b} (floor division)", a // b
        elif kind == "mod":
            a, b = num(rng, 25, 40), num(rng, 5, 15)
            text, v = f"{a} % {b} (the remainder)", a % b
        elif kind == "pow":
            base = rng.randint(3, 99)
            e = rng.randint(8, 40)
            text, v = f"{base} ** {e}", base ** e
        elif kind == "gcd":
            g = num(rng, 4, 10)
            a, b = g * num(rng, 6, 15), g * num(rng, 6, 15)
            text, v = f"gcd({a}, {b})", __import__("math").gcd(a, b)
        elif kind == "frac":
            a, b, c, d = (num(rng, 3, 9) for _ in range(4))
            text, v = f"{a}/{b} + {c}/{d} (as a reduced fraction)", Fraction(a, b) + Fraction(c, d)
        else:
            x, y = dec(rng, 3, 12), dec(rng, 3, 12)
            with localcontext() as ctx:
                ctx.prec = 200
                text, v = f"{x} * {y}", x * y
        if fits(v):
            return f"Compute {text} exactly.", fmt(v), {"kind": kind}


# ---- group 2: near the computer's limits --------------------------------------------------------------------------
def g2(rng):
    while True:
        kind = rng.choice(["long", "long", "bigpow"])
        if kind == "long":   # many fraction operations in one expression: may exhaust the firmware's registers
            terms = [f"{rng.randint(2, 999)}/{rng.randint(2, 99)}" for _ in range(rng.randint(5, 8))]
            ops = [rng.choice("+-*") for _ in terms[1:]]
            text = terms[0] + "".join(f" {o} ({t})" for o, t in zip(ops, terms[1:]))
            v = _evalfrac(text)
        else:                # a power whose final value fits 62 digits but whose steps may not, in the firmware
            base = rng.randint(2, 9)
            e = rng.randint(40, 62)
            text, v = f"{base} ** {e}", base ** e
        if fits(v):
            return f"Compute {text} exactly.", fmt(v), {"kind": kind}


def _evalfrac(text):
    """Exact value of a + - * / expression of integers and parentheses (our own generated text only)."""
    import ast
    ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b}

    def ev(n):
        if isinstance(n, ast.BinOp):
            return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return Fraction(n.value)
        raise ValueError(ast.dump(n))
    return ev(ast.parse(text, mode="eval").body)


# ---- group 3: word problems, 2-5 steps ---------------------------------------------------------------------------
def g3(rng):
    while True:
        t = rng.randrange(7)
        if t == 0:
            a, b, c, d = num(rng, 6, 12), num(rng, 4, 9), num(rng, 6, 12), num(rng, 2, 4)
            text = (f"A warehouse holds {a:,} crates with {b:,} units in each. {c:,} units are shipped out, and the "
                    f"rest are split equally among {d:,} trucks. How many units does each truck carry? Give the exact "
                    f"value, as a reduced fraction if it doesn't divide evenly.")
            v, steps = Fraction(a * b - c, d), 3
        elif t == 1:
            a, r, n, f = num(rng, 8, 14), num(rng, 4, 8), num(rng, 3, 5), num(rng, 6, 10)
            text = (f"A savings account starts with {a:,} cents. It earns {r:,} cents a day for {n:,} days, then a "
                    f"fee of {f:,} cents is taken. How many cents are in the account?")
            v, steps = a + r * n - f, 3
        elif t == 2:
            a, b, c = num(rng, 6, 12), num(rng, 6, 12), num(rng, 3, 6)
            text = (f"A field is {a:,} m long and {b:,} m wide. A square plot with sides of {c:,} m is fenced off "
                    f"inside it. What area, in square metres, is left?")
            v, steps = a * b - c * c, 3
        elif t == 3:
            p, h, d, w = num(rng, 4, 8), rng.randint(6, 23), num(rng, 3, 5), num(rng, 3, 7)
            text = (f"A factory makes {p:,} parts an hour, {h} hours a day, for {d:,} days. {w:,} parts are "
                    f"rejected. How many good parts are made?")
            v, steps = p * h * d - w, 3
        elif t == 4:
            a, k, y = num(rng, 5, 10), rng.randint(2, 9), rng.randint(5, 20)
            text = (f"A colony starts with {a:,} cells, and the number of cells multiplies by {k} every hour. How "
                    f"many cells are there after {y} hours?")
            v, steps = a * k ** y, 2
        elif t == 5:
            a, b, c, n = num(rng, 6, 12), num(rng, 6, 12), num(rng, 6, 12), rng.randint(3, 9)
            text = (f"Three shipments weigh {a:,} g, {b:,} g and {c:,} g. They are combined and divided equally "
                    f"into {n} parcels. What does each parcel weigh, in grams? Give the exact value, as a reduced "
                    f"fraction if it doesn't divide evenly.")
            v, steps = Fraction(a + b + c, n), 3
        else:
            s, n, pr, d = num(rng, 3, 6), num(rng, 5, 9), num(rng, 3, 7), num(rng, 3, 8)
            text = (f"A shop sells {n:,} items at {pr:,} cents each to each of {s:,} stores, then gives a total "
                    f"discount of {d:,} cents. What is the total revenue in cents?")
            v, steps = s * n * pr - d, 4
        if fits(v) and v > 0:                # a real-world quantity: never negative
            return text, fmt(v), {"template": t, "steps": steps}


# ---- group 4: dependent chains ------------------------------------------------------------------------------------
def g4(rng):
    while True:
        v = num(rng, 5, 8)
        lines, inter = [f"Start with {v:,}."], []
        n = rng.randint(8, 15)
        for i in range(1, n + 1):
            op = rng.choice(["mul", "mul", "add", "sub", "floordiv", "mod"])
            if op == "mul" or (op in ("floordiv", "mod") and v < 10 ** 6):
                k = rng.randint(101, 99999)
                v, s = v * k, f"multiply by {k:,}"
            elif op == "add":
                k = num(rng, 6, 12)
                v, s = v + k, f"add {k:,}"
            elif op == "sub":
                k = rng.randint(1, max(1, v // 2))
                v, s = v - k, f"subtract {k:,}"
            elif op == "floordiv":
                k = rng.randint(7, 9999)
                v, s = v // k, f"divide by {k:,} and drop the remainder"
            else:
                k = num(rng, 6, 9)
                v, s = v % k, f"replace the number with its remainder after dividing by {k:,}"
            lines.append(f"Step {i}: {s}.")
            inter.append(str(v))
        if all(len(x) <= 40 for x in inter) and v > 0:
            text = " ".join(lines) + " What is the final number?"
            return text, str(v), {"steps": n, "intermediates": inter}


# ---- group 5a: code with hidden tests (fixed pools; pilot and final pools never overlap) ---------------------------
CODE_POOL = {
    "pilot": [
        ("count_vowels", "s", "returns how many vowels (a, e, i, o, u, either case) the string s contains",
         [("'Hello World'",), ("'AEIOU xyz'",), ("''",)], ["3", "5", "0"]),
        ("second_largest", "nums", "returns the second largest distinct value in the list nums (None if there "
         "isn't one)", [("[3, 1, 4, 4, 2]",), ("[7]",), ("[5, 5]",), ("[-1, -3, -2]",)], ["3", "None", "None", "-2"]),
    ],
    "final": [
        ("digit_sum_factorial", "n", "returns the sum of the decimal digits of n! (n factorial)",
         [("10",), ("50",), ("0",), ("100",)], ["27", "216", "1", "648"]),
        ("run_length", "s", "returns the run-length encoding of s as a string, e.g. 'aaabcc' -> 'a3b1c2'",
         [("'aaabcc'",), ("''",), ("'z'",), ("'aabbaa'",)], ["'a3b1c2'", "''", "'z1'", "'a2b2a2'"]),
        ("to_roman", "n", "returns the Roman numeral for 1 <= n <= 3999 as an upper-case string",
         [("1994",), ("4",), ("3999",), ("58",)], ["'MCMXCIV'", "'IV'", "'MMMCMXCIX'", "'LVIII'"]),
        ("balanced", "s", "returns True if the brackets ()[]{} in s are balanced and properly nested, else False "
         "(other characters are ignored)", [("'([]{})'",), ("'([)]'",), ("'a(b)c'",), ("'(('",)],
         ["True", "False", "True", "False"]),
        ("merge_intervals", "intervals", "merges overlapping [start, end] intervals and returns them sorted by "
         "start, as a list of lists", [("[[1, 3], [2, 6], [8, 10]]",), ("[]",), ("[[1, 4], [4, 5]]",),
                                       ("[[5, 6], [1, 2]]",)],
         ["[[1, 6], [8, 10]]", "[]", "[[1, 5]]", "[[1, 2], [5, 6]]"]),
        ("nth_prime", "n", "returns the n-th prime number (nth_prime(1) == 2)",
         [("1",), ("10",), ("100",), ("1000",)], ["2", "29", "541", "7919"]),
        ("flatten", "xs", "flattens arbitrarily nested lists into one flat list, keeping order",
         [("[1, [2, [3, [4]], 5]]",), ("[]",), ("[[[]]]",), ("[[1, 2], 3]",)],
         ["[1, 2, 3, 4, 5]", "[]", "[]", "[1, 2, 3]"]),
        ("binary_to_int", "b", "converts a string of 0s and 1s to its integer value, without using int(b, 2)",
         [("'0'",), ("'101'",), ("'11111111'",), ("'1' + '0' * 70",)], ["0", "5", "255", str(2 ** 70)]),
        ("transpose", "m", "returns the transpose of the rectangular matrix m (a list of rows)",
         [("[[1, 2, 3], [4, 5, 6]]",), ("[[1]]",), ("[]",)], ["[[1, 4], [2, 5], [3, 6]]", "[[1]]", "[]"]),
        ("collatz_steps", "n", "returns how many steps the Collatz process takes to reach 1 from n (n -> n/2 if "
         "even, else 3n+1)", [("1",), ("6",), ("27",), ("97",)], ["0", "8", "111", "118"]),
    ],
}


def g5a(rng, pool, used):
    func, arg, what, args, outs = pool[len(used)]
    text = (f"Write a Python function `{func}({arg})` that {what}. Reply with the complete function in one "
            f"```python code block.")
    tests = [{"call": f"{func}({a[0]})", "expect": e} for a, e in zip(args, outs)]
    return text, None, {"func": func, "tests": tests}


# ---- group 5b: code reading --------------------------------------------------------------------------------------
READ_TEMPLATES = [
    "def f(n):\n    t = 0\n    for i in range(1, n + 1):\n        if i % {a} == 0:\n            t += i * {b}\n"
    "        else:\n            t -= {c}\n    return t",
    "def f(n):\n    a, b = {a}, {b}\n    for _ in range(n):\n        a, b = b, (a + b) % {c}\n    return a",
    "def f(n):\n    s = []\n    while n:\n        s.append(n % {a})\n        n //= {a}\n    return sum(d * {b} for d in s) + len(s) * {c}",
    "def f(n):\n    x = {a}\n    for i in range(n):\n        x = x * {b} - i if i % 2 else x + {c}\n    return x",
]


def g5b(rng):
    tpl = rng.choice(READ_TEMPLATES)
    a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(1, 50)
    src = tpl.format(a=a, b=b, c=c)
    n = rng.randint(10, 40)
    env = {}
    exec(src, env)                    # noqa: S102  (our own template; computes the expected answer)
    v = env["f"](n)
    text = f"What does f({n}) return for this Python function?\n\n```python\n{src}\n```"
    return text, str(v), {"template": READ_TEMPLATES.index(tpl)}


def main():
    which, out = sys.argv[1:3]
    rng = random.Random(SEEDS[which])
    rows, used5a = [], []
    for group, count in COUNTS[which].items():
        for i in range(count):
            if group == "5a":
                text, ans, meta = g5a(rng, CODE_POOL[which], used5a)
                used5a.append(meta["func"])
            else:
                text, ans, meta = {"1": g1, "2": g2, "3": g3, "4": g4, "5b": g5b}[group](rng)
            prompt = text + ("" if group == "5a" else FINAL)
            rows.append({"id": f"{which}-{group}-{i + 1:03d}", "group": group, "prompt": prompt, "answer": ans,
                         **meta})
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} questions -> {out}")


if __name__ == "__main__":
    main()
