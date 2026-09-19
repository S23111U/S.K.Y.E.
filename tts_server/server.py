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
        "tempo": 1.08, "pitch": 1.03, "gain": 1.0}}
       tempo: speaking-speed multiplier that leaves pitch and voice character
       untouched (WSOLA time-stretch); pitch: small pitch multiplier (keep
       within ~+-4%, or the voice stops sounding like the same person);
       gain: loudness multiplier.
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

class Stretcher:
    """Streaming WSOLA time-stretch: changes speaking speed while leaving pitch
    and formants (the voice's identity) alone. speed > 1 is faster.

    An earlier version changed pace by resampling, which also shifted every
    formant — each mood then sounded like a different person. Here 32 ms
    Hann-windowed frames are re-laid at a different hop, and each frame's
    source position is nudged (within +-8 ms) to the point that best continues
    the previous frame, so waveform periods line up and there is no warble.
    """

    def __init__(self, speed, sr=24000):
        self.speed = speed
        self.N = int(0.032 * sr)
        self.Hs = self.N // 2
        self.Ha = self.Hs * speed
        self.tol = int(0.008 * sr)
        self.win = np.hanning(self.N + 1)[:-1].astype(np.float32)  # sums to 1 at 50% overlap
        self.buf = np.zeros(0, np.float32)
        self.base = 0            # absolute input index of buf[0]
        self.k = 0
        self.prev = None         # absolute input position chosen for the previous frame
        self.acc = np.zeros(self.N, np.float32)

    def push(self, x):
        if self.speed == 1.0:
            return x
        self.buf = np.concatenate([self.buf, x])
        out = []
        N, Hs = self.N, self.Hs
        while True:
            nominal = int(round(self.k * self.Ha))
            need_to = max(nominal + self.tol, (self.prev or 0) + Hs) + N   # exclusive, absolute
            if need_to > self.base + len(self.buf):
                break
            if self.prev is None:
                pos = 0
            else:
                lo = max(nominal - self.tol, self.base)
                target = self.buf[self.prev + Hs - self.base : self.prev + Hs - self.base + N]
                best, best_score = lo, -np.inf
                # correlation of every candidate segment with the natural continuation
                cands = np.lib.stride_tricks.sliding_window_view(
                    self.buf[lo - self.base : nominal + self.tol - self.base + N], N
                )
                scores = cands @ target
                scores = scores / (np.linalg.norm(cands, axis=1) + 1e-9)
                best = lo + int(np.argmax(scores))
                pos = best
            seg = self.buf[pos - self.base : pos - self.base + N]
            self.acc += self.win * seg
            out.append(self.acc[:Hs].copy())
            self.acc = np.concatenate([self.acc[Hs:], np.zeros(Hs, np.float32)])
            self.prev = pos
            self.k += 1
            # drop input that can no longer be referenced
            keep_from = min(max(int(round(self.k * self.Ha)) - self.tol, 0), pos + Hs)
            if keep_from > self.base:
                self.buf = self.buf[keep_from - self.base :]
                self.base = keep_from
        return np.concatenate(out) if out else np.zeros(0, np.float32)


class Retimer:
    """Streaming playback-rate change (pitch and pace together). Used only for
    the tiny pitch component; linear interpolation with the read position
    carried across chunks so boundaries stay continuous."""

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
    # Net effect wanted: speed x tempo, pitch x pitch. Stretch by tempo/pitch,
    # then play back at `pitch` (which speeds up by pitch and raises it by pitch).
    tempo, pitch = float(params.get("tempo", 1.0)), float(params.get("pitch", 1.0))
    stretcher, retimer = Stretcher(tempo / pitch), Retimer(pitch)
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
            audio = retimer.push(stretcher.push(np.array(r.audio, dtype=np.float32).reshape(-1)))
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
