"""EARLYSTOP-PLAN v2: the frozen early-stop rule (chosen on development data, results/earlystop-dev.md).

RepetitionStop(k=64, r=6) watches only the text tokens the language model chooses (positions where the circuit
doesn't own the next token); never chip tokens or the circuit's forced echoes. It's consulted after each such token
is decided and before it's appended: `refuse(token)` returns True if appending it would complete the 6th occurrence
of the same 64-token span among the model's own tokens. Then generation ends without that token, and the answer counts
as a failure. EOS is never refused: if the chosen token is EOS, the answer completes normally. The same code serves the
offline evaluation and the live run.
"""

K, R = 64, 6
EOS = (248046, 248044)


class RepetitionStop:
    def __init__(self, k=K, r=R):
        self.k, self.r = k, r
        self.own, self.seen = [], {}

    def refuse(self, token):
        """Called with each Qwen-chosen next token before it's appended. True = stop here (the token isn't added)."""
        if token in EOS:
            return False
        self.own.append(token)
        if len(self.own) >= self.k:
            key = tuple(self.own[-self.k:])
            n = self.seen.get(key, 0) + 1
            if n >= self.r:
                self.own.pop()
                return True
            self.seen[key] = n
        return False


def stop_index(own_tokens, k=K, r=R):
    """Offline: the index in the model's own token sequence of the first refused token, or None."""
    rule = RepetitionStop(k, r)
    for i, t in enumerate(own_tokens):
        if rule.refuse(t):
            return i
    return None
