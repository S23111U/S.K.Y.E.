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

Inbound (client -> server) frames
----------------------------------
  {"type": "text",  "text": "..."}                                  typed input, or a client-confirmed transcript — triggers the LLM
  {"type": "audio", "pcm": "<base64 PCM>", "sample_rate": 16000}     a captured utterance (int16 mono PCM) — transcribed only, does NOT trigger the LLM

Every inbound message must be one of the above — there is no bare/unframed
text input anymore. An "audio" frame gets exactly one "transcript" frame
back (see below); the client decides what to do with it (wake-word/sleep/
cancel/filler filtering, etc.) and, if it should be acted on, sends the
cleaned text back as a "text" frame — that is what actually invokes the LLM.

Outbound: transcript echo
--------------------------
  {"type": "transcript", "text": "..."}   the raw Whisper transcript of an inbound "audio" frame

Outbound audio frames
----------------------
  {"type": "audio_chunk", "pcm": "<base64 f32le mono PCM>", "sample_rate": 24000, "final": false}
  {"type": "turn_end"}

"audio_chunk" frames (if any) follow the "done" frame, one per synthesized
speech chunk; "turn_end" always closes out a turn and is what clients should
watch for instead of "done" if they expect audio to follow.

Outbound: proactive (unsolicited) turns
-----------------------------------------
  {"type": "proactive", "text": "..."}   SKYE speaking first — a due task/reminder firing, not a reply to anything the client sent

Sent with no inbound frame having triggered it — the server's own scheduler
pushes this the moment a task comes due, to every currently-connected
client. Followed by the same "audio_chunk"/"turn_end" frames as a normal
turn. Requires the transport to read continuously rather than only right
after forwarding a client message (see clients/browser.py's bridge).
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
