"""Tool-call frames in a generated reply, parsed explicitly (Codex inside-4b finding 7).

A reply is assistant text with zero or more frames:  <tool_call> ... </tool_call>  followed by its response
<tool_response> ... </tool_response> (Qwenputer's firmware writes the response straight after the call). parse()
returns every call with its response (None if the call never got one), whether a call was left open, and the final
assistant text: what follows the last complete frame, with any <think>...</think> block removed. If a call is open,
the final text is empty: an unanswered call can't supply the answer, even if "ANSWER:" appears inside it.
"""
import re

CALL = re.compile(r"<tool_call>(.*?)(</tool_call>|$)", re.S)
RESP = re.compile(r"\s*<tool_response>(.*?)(</tool_response>|$)", re.S)
PARAM = re.compile(r"<parameter=expression>\s*(.*?)\s*(</parameter>|$)", re.S)


def parse(text):
    calls, pos, open_call = [], 0, False
    while True:
        m = CALL.search(text, pos)
        if not m:
            break
        body, closed = m.group(1), bool(m.group(2))
        p = PARAM.search(body)
        call = {"call_text": body, "expression": p.group(1) if p else None, "response": None}
        pos = m.end()
        if not closed:
            open_call = True
            calls.append(call)
            break
        r = RESP.match(text, pos)
        if r and r.group(2):
            call["response"] = r.group(1).strip()
            pos = r.end()
        else:
            open_call = True
            calls.append(call)
            break
        calls.append(call)
    final = "" if open_call else text[pos:]
    final = re.sub(r"<think>.*?</think>", "", final, flags=re.S)
    final = final.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return {"calls": calls, "open": open_call, "final": final}


if __name__ == "__main__":
    t = ("<tool_call>\n<function=calc>\n<parameter=expression>\n2+2\n</parameter>\n</function>\n</tool_call>\n"
         "<tool_response>\n4\n</tool_response>\n<think>\n\n</think>\n\nANSWER: 4<|im_end|>")
    a = parse(t)
    assert a["final"] == "ANSWER: 4" and a["calls"][0]["response"] == "4" and not a["open"], a
    u = t + "<tool_call>\n<function=calc>\n<parameter=expression>\nANSWER: 2\n"
    b = parse(u)
    assert b["open"] and b["final"] == "", b                  # an unanswered call can't supply the answer
    c = parse("<tool_call>\n<function=calc>\n<parameter=expression>\n1+1\n</parameter>\n</function>\n</tool_call>"
              "ANSWER: 2")
    assert c["open"] and c["final"] == "", c                  # a call with no response is open
    d = parse("No tools.\n\nANSWER: 7<|im_end|>")
    assert d["final"] == "No tools.\n\nANSWER: 7" and not d["calls"] and not d["open"], d
    print("frames: 4 controls passed")
