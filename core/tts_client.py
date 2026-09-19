"""Client for tts_server/ (Chatterbox Turbo, voice-cloned from the user's own recording).

Replaces the in-process XTTS engine (core/tts.py, now deleted). Turbo needs a
newer mlx than the pinned runtime the LLM and Whisper use, so it runs as a
subprocess under the repo's .venv and this module talks to it over line-
delimited JSON on stdin/stdout — see tts_server/server.py for the protocol.

Same call shape the rest of core_v2.py already used:
`synthesize_reply(text, ...)` yields (pcm_float32_bytes, is_final) at 24 kHz
mono. The difference is that audio now streams *while it is generated*
instead of arriving one finished sentence at a time, and cancellation takes
effect within a chunk (~1.5 s of audio) rather than after a whole sentence.
"""

import base64
import hashlib
import itertools
import json
import os
import queue
import subprocess
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTS_PYTHON = os.environ.get("SKYE_TTS_PYTHON", os.path.join(ROOT, ".venv", "bin", "python"))
SERVER_SCRIPT = os.path.join(ROOT, "tts_server", "server.py")
# The 12.8 s slice of assets/reference_voice.wav the engine clones from
# (Turbo wants roughly 6-15 s; regenerate this file if the reference changes).
REFERENCE_VOICE = os.path.join(ROOT, "assets", "reference_voice_short.wav")
FILLER_CACHE_DIR = os.path.join(ROOT, "assets", "filler_cache")
SAMPLE_RATE = 24000
STARTUP_TIMEOUT_S = 600  # first run downloads the model

# Terminates a turn's audio: browser.html marks speech as finished on the
# chunk flagged final, but streaming only learns a chunk was the last one
# after the fact, so the end is signalled with a few ms of silence.
_END_MARKER = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32).tobytes()

_ids = itertools.count(1)
_lock = threading.Lock()
_proc = None
_ready = threading.Event()
_pending: dict[int, queue.Queue] = {}
_write_lock = threading.Lock()


def _reader(proc):
    for line in proc.stdout:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("ready"):
            _ready.set()
            continue
        q = _pending.get(msg.get("id"))
        if q is not None:
            q.put(msg)
    # Process died: unblock anything still waiting on it.
    for q in list(_pending.values()):
        q.put({"end": True, "cancelled": False, "error": "tts server exited"})


def start():
    """Spawns the TTS server (idempotent) and blocks until it has loaded."""
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None:
            return
        _ready.clear()
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        log = open(os.path.join(ROOT, "logs", "tts_server.log"), "ab")
        _proc = subprocess.Popen(
            [TTS_PYTHON, SERVER_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
        )
        threading.Thread(target=_reader, args=(_proc,), daemon=True).start()
    if not _ready.wait(STARTUP_TIMEOUT_S):
        raise RuntimeError("TTS server did not become ready (see logs/tts_server.log)")


def _send(obj):
    with _write_lock:
        _proc.stdin.write(json.dumps(obj) + "\n")
        _proc.stdin.flush()


def synthesize_reply(
    text: str,
    *,
    params: dict | None = None,
    cancel: threading.Event | None = None,
    timings: dict | None = None,
):
    """Yields (pcm_float32_bytes, is_final) as audio is generated.

    `params`: sampling settings for the mood (see core/mood.py), plus an
    optional "voice" naming a per-mood reference clip.
    `cancel`: when set (the user barged in) generation is stopped server-side
    and nothing further is yielded.
    `timings` (dict, optional) receives first_chunk_ms and total_ms.
    """
    text = text.strip()
    if not text:
        return
    start()
    rid = next(_ids)
    q: queue.Queue = queue.Queue()
    _pending[rid] = q
    finished = False
    try:
        _send({"op": "say", "id": rid, "text": text, "params": params or {}})
        cancel_sent = False
        while True:
            try:
                msg = q.get(timeout=0.05)
            except queue.Empty:
                if cancel is not None and cancel.is_set() and not cancel_sent:
                    _send({"op": "cancel", "id": rid})
                    cancel_sent = True
                continue
            if msg.get("end"):
                finished = True
                if timings is not None:
                    timings["first_chunk_ms"] = msg.get("first_ms")
                    timings["total_ms"] = msg.get("gen_ms")
                if msg.get("error"):
                    raise RuntimeError(msg["error"])
                if not msg.get("cancelled") and not (cancel is not None and cancel.is_set()):
                    yield _END_MARKER, True
                return
            if cancel is not None and cancel.is_set():
                if not cancel_sent:
                    _send({"op": "cancel", "id": rid})
                    cancel_sent = True
                continue  # discard audio produced before the cancel landed
            yield base64.b64decode(msg["pcm"]), False
    finally:
        # Consumer stopped iterating early (or an error): don't leave the
        # server generating audio nobody will hear.
        if not finished:
            try:
                _send({"op": "cancel", "id": rid})
            except Exception:
                pass
        _pending.pop(rid, None)


def cached_phrase(phrase: str, mood: str = "calm") -> bytes:
    """Audio for a fixed short phrase, synthesized once and kept on disk.

    Fillers cover the LLM's thinking time; synthesizing them live would put
    TTS on the GPU at exactly the moment the LLM needs it. Keyed on the phrase,
    mood settings, engine and the reference clip's mtime, so changing the
    voice or a mood's parameters regenerates. The mood shapes the delivery (an empathetic filler is spoken
    softly, a pleased one brightly).
    """
    from core.mood import MOOD_PARAMS  # local: keeps this module importable alone

    key = hashlib.md5(
        f"turbo|{mood}|{sorted(MOOD_PARAMS[mood].items())}|{phrase}|{os.path.getmtime(REFERENCE_VOICE)}".encode()
    ).hexdigest()[:16]
    path = os.path.join(FILLER_CACHE_DIR, f"{key}.f32")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return _level(f.read())
    pcm = b"".join(
        p for p, final in synthesize_reply(phrase, params=MOOD_PARAMS[mood]) if not final
    )
    os.makedirs(FILLER_CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        f.write(pcm)
    return _level(pcm)


def _level(pcm: bytes, target_rms: float = 0.035) -> bytes:
    """Brings a cached phrase to a consistent loudness. Turbo's level varies a
    lot between short phrases (one filler came out ~4x quieter than the rest),
    and a filler nobody can hear defeats its purpose."""
    a = np.frombuffer(pcm, dtype=np.float32)
    rms = float(np.sqrt((a ** 2).mean())) if len(a) else 0.0
    if rms < 1e-6:
        return pcm
    gain = min(target_rms / rms, 0.9 / max(float(np.abs(a).max()), 1e-6))
    return (a * gain).astype(np.float32).tobytes()
