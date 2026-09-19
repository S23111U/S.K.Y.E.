"""SKYE's text-to-speech server: Chatterbox Turbo (mlx-audio), voice-cloned.

Runs as its own subprocess (spawned by core/tts_client.py) under the repo's
.venv, because mlx-audio needs a newer mlx than the pinned runtime the LLM
and Whisper run on — the same isolation idea as mcp_server/.

Protocol: one JSON object per line. stdin carries requests, stdout carries
ONLY protocol messages (everything else — model loading chatter, warnings —
is pushed to stderr; a stray stdout line would corrupt the stream, the same
trap the MCP server hit).

  in : {"op": "say", "id": 3, "text": "...", "params": {"temperature": 0.8,
        "top_p": 0.95, "repetition_penalty": 1.2, "voice": "happy",
        "rate": 1.07, "gain": 1.0}}
       rate: playback-rate change applied to the output (pitch and pace move
       together, like real emotional speech); gain: loudness multiplier.
       {"op": "cancel", "id": 3}
  out: {"ready": true}                                  once, after warm-up
       {"id": 3, "pcm": "<b64 f32le mono>", "sr": 24000}  as audio is produced
       {"id": 3, "end": true, "first_ms": 410, "gen_ms": 2210, "cancelled": false}

Audio is streamed as it is generated (not per finished sentence), and
cancellation is checked between chunks, so a barge-in stops speech within a
fraction of a second.
"""

import base64
import json
import os
import queue
import sys
import threading
import time

_real_stdout = sys.stdout
sys.stdout = sys.stderr

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
REPO = "mlx-community/chatterbox-turbo-fp16"
DEFAULT_REF = os.path.join(ASSETS, "reference_voice_short.wav")
STREAM_INTERVAL_S = 1.5  # audio per streamed chunk: lower = earlier first word, slower overall

_send_lock = threading.Lock()


def send(obj):
    with _send_lock:
        _real_stdout.write(json.dumps(obj) + "\n")
        _real_stdout.flush()


model = load_model(REPO)

# Conditioning is computed once per voice and swapped in per request, instead
# of re-deriving it from the reference clip on every sentence. "default" is
# the user's own reference voice; optional per-mood clips
# (assets/reference_voice_<mood>.wav, e.g. recorded in a happy/sad tone)
# override it for that mood when present.
VOICES = {}
model.prepare_conditionals(DEFAULT_REF)
VOICES["default"] = model._conds
for mood in ("happy", "sad", "calm", "concerned"):
    path = os.path.join(ASSETS, f"reference_voice_{mood}.wav")
    if os.path.isfile(path):
        model.prepare_conditionals(path)
        VOICES[mood] = model._conds
model._conds = VOICES["default"]
print(f"[tts_server] voices: {sorted(VOICES)}", file=sys.stderr)

for _ in model.generate(text="Warm up.", verbose=False):
    pass

class Retimer:
    """Streaming playback-rate change (rate > 1: faster and higher; < 1: slower
    and lower). Linear interpolation with the read position carried across
    chunks, so chunk boundaries stay continuous (no clicks)."""

    def __init__(self, rate):
        self.rate, self.buf, self.pos = rate, np.zeros(0, np.float32), 0.0

    def push(self, x):
        if self.rate == 1.0:
            return x
        self.buf = np.concatenate([self.buf, x])
        idx = np.arange(self.pos, len(self.buf) - 1, self.rate)
        if len(idx) == 0:
            return np.zeros(0, np.float32)
        i0 = idx.astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        out = self.buf[i0] * (1 - frac) + self.buf[i0 + 1] * frac
        nxt = idx[-1] + self.rate
        drop = int(nxt)
        self.buf, self.pos = self.buf[drop:], nxt - drop
        return out


requests = queue.Queue()
cancelled = set()
cancel_lock = threading.Lock()


def reader():
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("op") == "cancel":
            with cancel_lock:
                cancelled.add(msg["id"])
        elif msg.get("op") == "say":
            requests.put(msg)
    requests.put(None)  # stdin closed: parent is gone


threading.Thread(target=reader, daemon=True).start()
send({"ready": True})

while True:
    req = requests.get()
    if req is None:
        break
    rid = req["id"]
    with cancel_lock:
        if rid in cancelled:
            cancelled.discard(rid)
            send({"id": rid, "end": True, "cancelled": True, "first_ms": None, "gen_ms": 0})
            continue

    params = req.get("params") or {}
    model._conds = VOICES.get(params.get("voice"), VOICES["default"])
    t0, first_ms, was_cancelled = time.time(), None, False
    retimer = Retimer(float(params.get("rate", 1.0)))
    gain = float(params.get("gain", 1.0))
    try:
        for r in model.generate(
            text=req["text"],
            verbose=False,
            stream=True,
            streaming_interval=STREAM_INTERVAL_S,
            temperature=params.get("temperature", 0.8),
            top_p=params.get("top_p", 0.95),
            repetition_penalty=params.get("repetition_penalty", 1.2),
        ):
            with cancel_lock:
                if rid in cancelled:
                    cancelled.discard(rid)
                    was_cancelled = True
                    break
            audio = retimer.push(np.array(r.audio, dtype=np.float32).reshape(-1))
            if gain != 1.0:
                audio = np.clip(audio * gain, -1.0, 1.0)
            if len(audio) == 0:
                continue
            if first_ms is None:
                first_ms = round((time.time() - t0) * 1000)
            send({"id": rid, "pcm": base64.b64encode(audio.tobytes()).decode("ascii"), "sr": r.sample_rate})
    except Exception as e:
        print(f"[tts_server] generation error: {type(e).__name__}: {e}", file=sys.stderr)
    send({
        "id": rid, "end": True, "cancelled": was_cancelled,
        "first_ms": first_ms, "gen_ms": round((time.time() - t0) * 1000),
    })
