"""The Python side of qwen_half_helper's pipe (see qwen_half_helper.cpp for the protocol), for the runners. Fails
closed: a helper error, a short read (the child died or closed its output), a request that doesn't finish within the
timeout (one deadline per request, covering the send of all its rows and the whole reply), or a bad reply (ids
outside the vocabulary or repeated, non-finite values, values not in descending order) raises HelperError. close()
always ends the child (END, sent with its own bound, then kill after a wait). As a context manager, a nonzero helper
exit on a normal exit also raises."""
import math, os, select, struct, subprocess, time

import numpy as np


class HelperError(RuntimeError):
    pass


class Helper:
    def __init__(self, exe, gguf, n_ctx=16384, stderr=None, n_vocab=None, timeout=300.0):
        self.p = subprocess.Popen([exe, gguf, str(n_ctx)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=stderr, bufsize=0)
        os.set_blocking(self.p.stdin.fileno(), False)      # writes wait on select with a deadline, never forever
        self.n_vocab, self.timeout = n_vocab, timeout
        self.deadline = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        rc = self.close()
        if exc_type is None and rc != 0:
            raise HelperError(f"helper exited with code {rc}")

    def _left(self, what, done, n):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise HelperError(f"helper request not done within {self.timeout:.0f} s ({what} {done} of {n} bytes)")
        return left

    def _read(self, n):
        buf, fd = b"", self.p.stdout.fileno()
        while len(buf) < n:
            if not select.select([fd], [], [], self._left("read", len(buf), n))[0]:
                continue                                  # the next _left() raises once the deadline has passed
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                raise HelperError(f"helper output ended (read {len(buf)} of {n} bytes; exit code {self.p.poll()})")
            buf += chunk
        return buf

    def _send(self, data):
        """All of data before the deadline; short writes are continued."""
        fd, view, done = self.p.stdin.fileno(), memoryview(data), 0
        while done < len(data):
            if not select.select([], [fd], [], self._left("sent", done, len(data)))[1]:
                continue
            try:
                done += os.write(fd, view[done:])
            except BlockingIOError:
                continue
            except OSError as e:                          # BrokenPipeError included: the child is gone
                raise HelperError(f"helper input closed ({e}; exit code {self.p.poll()})") from e

    def _reply(self, want_ids):
        st = struct.unpack("<i", self._read(4))[0]
        if st != 0:
            n = struct.unpack("<I", self._read(4))[0]
            raise HelperError("helper: " + self._read(n).decode(errors="replace"))
        if not want_ids:
            return None
        ids = list(struct.unpack("<5i", self._read(20)))
        vals = list(struct.unpack("<5f", self._read(20)))
        hi = self.n_vocab if self.n_vocab is not None else 2**31
        if not all(0 <= t < hi for t in ids) or len(set(ids)) != 5:
            raise HelperError(f"bad reply ids {ids}")
        if not all(math.isfinite(v) for v in vals) or any(a < b for a, b in zip(vals, vals[1:])):
            raise HelperError(f"bad reply values {vals}")
        return ids, vals

    def _request(self, data, want_ids):
        self.deadline = time.monotonic() + self.timeout
        self._send(data)
        return self._reply(want_ids)

    def reset(self):
        self._request(struct.pack("<II", 1, 0), False)

    def tokens(self, pos, ids):
        """Feed token ids at text positions; returns (top-5 ids, values) after the last; ids[0] = the argmax (the
        lowest id at a tie)."""
        return self._request(struct.pack("<II", 4, len(pos)) + np.asarray(pos, "<i4").tobytes()
                             + np.asarray(ids, "<i4").tobytes(), True)

    def rows(self, pos, rows):
        rows = np.ascontiguousarray(rows, dtype="<f4")
        return self._request(struct.pack("<II", 2, len(pos)) + np.asarray(pos, "<i4").tobytes() + rows.tobytes(), True)

    def close(self, wait=30.0):
        if self.p.poll() is None:
            try:
                self.deadline = time.monotonic() + 5.0       # END is sent with a bound too
                self._send(struct.pack("<II", 3, 0))
                self.p.wait(wait)
            except (HelperError, subprocess.TimeoutExpired):
                pass
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait(10)
        return self.p.returncode
