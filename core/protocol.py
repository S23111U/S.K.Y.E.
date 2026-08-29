"""
SKYE wire protocol.

One JSON object per line ("NDJSON"). Newline is a safe delimiter because JSON
escapes literal newlines inside strings as \\n, so a frame boundary can never
appear inside a payload. This replaces the old "..." delimiter, which appeared
in ordinary prose and silently ate ellipses and full stops.

Frame types
-----------
  {"type": "start"}                       generation began
  {"type": "token",  "text": "..."}       one chunk of the reply, as generated
  {"type": "tool",   "name": "tell_time"} a tool is being executed
  {"type": "done",   "text": "..."}       final CLEANED reply, guardrails applied
  {"type": "error",  "message": "..."}    something failed

Clients that cannot stream may ignore every frame except "done", whose text is
always the complete, sanitised reply. Clients that can stream should render
"token" frames live and then replace that text with the "done" text, because
guardrails only run once the full response exists.
"""

import json


def frame(kind: str, **fields) -> bytes:
    """Encode one frame, newline-terminated, ready to send."""
    return (json.dumps({"type": kind, **fields}, ensure_ascii=False) + "\n").encode("utf-8")


class FrameReader:
    """Accumulates bytes from a socket and yields complete frames.

    TCP does not preserve message boundaries: one recv() may return half a
    frame, or three frames at once. This buffers until a newline is seen and
    keeps any partial remainder for the next call.
    """

    def __init__(self):
        self._buf = b""

    def feed(self, chunk: bytes):
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                yield json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                yield {"type": "error", "message": "malformed frame"}
