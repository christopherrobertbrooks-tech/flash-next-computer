"""The calc tool's accepted-input contract, shared by both MCP backends (the computer and host Python), so stage 5
compares them on the same inputs. An expression is refused, with nothing run, if it is not a string, contains a
protocol marker, encodes to a special token, or makes a call longer than the circuit's capture buffer
(spec.CAP_MAX tokens between <tool_call> and </tool_call>). Needs only the Qwenputer tokenizer (no GPU, no weights)."""
from chipformer.rv import spec as S

CALL = "<tool_call>\n<function=calc>\n<parameter=expression>\n{expr}\n</parameter>\n</function>\n</tool_call>"
MARKERS = ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<function", "</function",
           "<parameter", "</parameter", "<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")


class CallRefused(Exception):
    """The call was not run (invalid session or input)."""


def first_special_id(tok):
    return min(i for i in tok.all_special_ids if i > 1000)            # Qwen's specials start here


def check_input(expr, tok, T, first_special):
    """The call's token ids, or CallRefused. Changes nothing."""
    if not isinstance(expr, str):
        raise CallRefused("the expression must be a string")
    for mk in MARKERS:
        if mk in expr:
            raise CallRefused(f"the expression contains the protocol marker {mk!r}")
    inner = tok.encode(expr, add_special_tokens=False)
    if any(i >= first_special for i in inner):
        raise CallRefused("the expression encodes to a special token")
    ids = tok.encode(CALL.format(expr=expr), add_special_tokens=False)
    if ids.count(T.TCO) != 1 or ids.count(T.TCC) != 1 or ids[-1] != T.TCC:
        raise CallRefused("the call must contain exactly one <tool_call> ... </tool_call>")
    captured = len(ids) - ids.index(T.TCO) - 2
    if captured > S.CAP_MAX:
        raise CallRefused(f"the call is {captured} tokens; the circuit captures at most {S.CAP_MAX}")
    return ids
