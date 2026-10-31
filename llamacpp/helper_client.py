"""The Python side of qwen_half_helper's pipe (see qwen_half_helper.cpp for the protocol), for the runners. Fails
closed: a helper error, a short read (the child died or closed its output) or a bad reply raises HelperError; close()
always ends the child (END, then kill after a timeout). Use it as a context manager."""
import struct, subprocess

import numpy as np


class HelperError(RuntimeError):
    pass


class Helper:
    def __init__(self, exe, gguf, n_ctx=16384, stderr=None):
        self.p = subprocess.Popen([exe, gguf, str(n_ctx)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=stderr)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _read(self, n):
        b = self.p.stdout.read(n)
        if b is None or len(b) != n:
            raise HelperError(f"helper output ended (read {0 if b is None else len(b)} of {n} bytes; "
                              f"exit code {self.p.poll()})")
        return b

    def _send(self, data):
        try:
            self.p.stdin.write(data)
            self.p.stdin.flush()
        except (BrokenPipeError, OSError) as e:
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
        return ids, vals

    def reset(self):
        self._send(struct.pack("<II", 1, 0))
        self._reply(False)

    def tokens(self, pos, ids):
        """Feed token ids at text positions; returns (top-5 ids, values) after the last; ids[0] = the argmax (the
        lowest id at a tie)."""
        self._send(struct.pack("<II", 4, len(pos)) + np.asarray(pos, "<i4").tobytes() + np.asarray(ids, "<i4").tobytes())
        return self._reply(True)

    def rows(self, pos, rows):
        rows = np.ascontiguousarray(rows, dtype="<f4")
        self._send(struct.pack("<II", 2, len(pos)) + np.asarray(pos, "<i4").tobytes() + rows.tobytes())
        return self._reply(True)

    def close(self):
        if self.p.poll() is None:
            try:
                self.p.stdin.write(struct.pack("<II", 3, 0))
                self.p.stdin.flush()
                self.p.wait(30)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                pass
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait(10)
        return self.p.returncode
