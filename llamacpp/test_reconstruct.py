"""CPU controls for test_split_tokens_gpu.reconstruct (Codex fast build review, finding 6), on a real reference
stream (a calc call, its run, the answer, EOS):
- the complete EOS stream reconstructs to its text inputs;
- budget cuts right after </tool_call>, inside the chip span and right after a forced BYTE echo are accepted (a
  matching prefix of the reference's run) and return exactly the text tokens processed;
- a mutated chip token, a mutated forced echo and a missing HALT newline are rejected, with or without a budget stop;
- a cut stream without a budget stop is rejected.
usage (riscv dir on PYTHONPATH, the qwenputer venv): python test_reconstruct.py QWENPUTER_MODEL_DIR
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import test_split_tokens_gpu as TG  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

V0 = 248320
CALL = "<tool_call>\\n<function=calc>\\n<parameter=expression>\\n84/18\\n</parameter>\\n</function>\\n</tool_call>"


def main():
    T = BR.Text(Path(sys.argv[1]) / "tokenizer.json")
    prompt = T.encode("<|im_start|>user\\nWhat is it?<|im_end|>\\n<|im_start|>assistant\\n")
    ses = BR.Session(T, TG.parked())
    ses.history(prompt)
    k = len(ses.toks)
    ses.qwen(T.encode(CALL.replace("\\\\n", "\\n")) + T.encode("ANSWER: 14/3"))
    gen = [t - S.TEXT_TOKEN if S.is_text(t) else V0 + t for t in ses.toks[k:]] + [248046]
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    def ok(g, budget):
        try:
            return TG.reconstruct(T, V0, prompt, g, budget_stop=budget)
        except AssertionError as e:
            return f"rejected: {e}"

    text = lambda g: [t for t in g if t < V0 and t != 248046]
    full = ok(gen, False)
    check(f"complete EOS stream: {len(full) if isinstance(full, list) else full} text inputs", full == text(gen))
    tcc = gen.index(T.TCC)
    chips = [i for i, t in enumerate(gen) if t >= V0]
    echo = next(i for i in range(chips[0], len(gen)) if gen[i] < V0)        # the first forced echo in the run
    for label, cut in (("right after </tool_call>", tcc + 1), ("inside the chip span", chips[0] + 3),
                       ("right after a forced echo", echo + 1)):
        g = gen[:cut]
        check(f"a budget cut {label} is accepted", ok(g, True) == text(g))
        check(f"  the same cut without a budget stop is rejected", isinstance(ok(g, False), str))
    for label, i, new in (("a mutated chip token", chips[5], gen[chips[5]] + 1 if gen[chips[5]] + 1 < V0 + S.VOCAB
                           else gen[chips[5]] - 1), ("a mutated forced echo", echo, (gen[echo] + 1) % V0)):
        for budget in (False, True):
            g = list(gen) if not budget else list(gen[:echo + 5])
            g[i] = new
            check(f"{label} is rejected (budget stop {budget})", isinstance(ok(g, budget), str))
    halt_nl = next(i for i in range(len(gen) - 1) if gen[i] == V0 + S.tok("HALT")) + 1
    g = gen[:halt_nl] + gen[halt_nl + 1:]
    check("a missing HALT newline is rejected", isinstance(ok(g, False), str) and isinstance(ok(g, True), str))
    print("RECONSTRUCT CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
