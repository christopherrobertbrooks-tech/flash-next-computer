"""CPU-only controls for rvlive.Shadow (no model, no GPU): feed it token streams built by upstream's reference Session
(exactly what a correct circuit produces) and broken copies of them. Exits 1 if any control fails.

usage (riscv dir on PYTHONPATH): python test_rvlive.py QWENPUTER_MODEL_DIR
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rvlive as L  # noqa: E402
from chipformer.rv import bridge as BR  # noqa: E402
from chipformer.rv import elf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

CALL = "<tool_call>\n<function=calc>\n<parameter=expression>\n{e}\n</parameter>\n</function>\n</tool_call>"


def stream(T, prompt, pieces):
    """The stream a correct circuit gives: model text pieces and calls, run through upstream's reference Session."""
    m = ref.Machine.from_program(elf.program(L.FIRMWARE))
    BR.park(m)
    sess = BR.Session(T, m)
    sess.history(prompt)
    k = len(sess.toks)
    for kind, text in pieces:
        sess.qwen(T.encode(CALL.format(e=text) if kind == "call" else text))
    return sess.toks[k:]


def main():
    T = BR.Text(Path(sys.argv[1]) / "tokenizer.json")
    prompt = T.encode("<|im_start|>user\nWhat is it?<|im_end|>\n<|im_start|>assistant\n")
    bad = 0

    def check(label, cond):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {label}", flush=True)

    # a correct stream: one call
    toks = stream(T, prompt, [("text", "Let me compute. "), ("call", "123456789*987654321"), ("text", "ANSWER: 1")])
    sh = L.Shadow(T, prompt, log_path=None)
    commits = sum(sh.feed(t) == "committed" for t in toks)
    check(f"one call: verified, {commits} records committed, {sh.tokens_checked} tokens checked", sh.verified and commits > 1000)
    check(f"one call: the shadow printed {bytes(sh.m.out)!r}", b"121932631112635269" in bytes(sh.m.out))
    check("one call: back to text after the call", not sh.in_call and sh.calls == 1)

    # two calls in one answer (the machine parks between them)
    toks = stream(T, prompt, [("call", "2+2"), ("text", "and "), ("call", "(2**64 - 1) // 7"), ("text", " done")])
    sh = L.Shadow(T, prompt)
    for t in toks:
        sh.feed(t)
    check(f"two calls: verified, output {bytes(sh.m.out)!r}", sh.verified and sh.calls == 2
          and b"4" in bytes(sh.m.out) and b"2635249153387078802" in bytes(sh.m.out))

    # one wrong computer token in the middle: flagged at that record, the shadow stops, the text continues
    toks = stream(T, prompt, [("call", "123456789*987654321"), ("text", "ANSWER: 1")])
    chips = [i for i, t in enumerate(toks) if not S.is_text(t)]
    i = chips[len(chips) // 2]
    broken = list(toks)
    broken[i] = toks[i] + 1 if toks[i] + 1 < S.VOCAB else toks[i] - 1
    log = Path("/tmp/rvlive-test-mismatch.jsonl")
    log.unlink(missing_ok=True)
    sh = L.Shadow(T, prompt, log_path=str(log))
    seen_at = None
    for n, t in enumerate(broken):
        sh.feed(t)
        if not sh.verified and seen_at is None:
            seen_at, rec_at = n, sh.records
    check(f"mismatch: flagged at the changed token (position {seen_at}, changed {i})", seen_at == i)
    check(f"mismatch: the shadow stopped ({sh.records} records, frozen at {rec_at})", sh.records == rec_at)
    check(f"mismatch: status says UNVERIFIED ({sh.status()[:40]}...)", sh.status().startswith("UNVERIFIED since record"))
    check("mismatch: logged", log.exists() and log.read_text().count("\n") == 1)
    check("mismatch: the chat text still streams", T.decode_bytes(sh.text_ids).decode().endswith("ANSWER: 1"))

    # a computer token where the model should be writing text
    sh = L.Shadow(T, prompt)
    sh.feed(BR.text_token(T.encode("Hi")[0]))
    sh.feed(S.tok("GO"))
    check("a computer token outside a call is flagged", not sh.verified and "outside" in sh.mismatch["why"])

    # the screen
    toks = stream(T, prompt, [("call", "84/18")])
    sh = L.Shadow(T, prompt)
    for t in toks[: len(toks) // 2]:
        sh.feed(t)
    sh.note = sh.status()
    lines = L.V.text_of(L.V.render(sh, 30, 120, 2), 120)
    check("screen: shadow label and VERIFIED status visible", any("SHADOW" in x for x in lines)
          and any("VERIFIED" in x for x in lines))
    print("LIVE CONTROLS", "PASSED" if not bad else f"FAILED ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
