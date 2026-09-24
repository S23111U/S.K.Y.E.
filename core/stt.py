"""Local speech-to-text via mlx-whisper.

Runs entirely on-device (Metal via MLX) so voice input never leaves the
machine, unlike the browser's SpeechRecognition (which streams raw mic
audio to Apple/Google's cloud even though the rest of SKYE is offline).
"""

import json
import os
import threading
import time
import wave
from collections import deque

import numpy as np
import mlx_whisper

WHISPER_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
STT_LOCK = threading.Lock()

print("Loading Whisper (mlx-whisper)...")
# mlx_whisper.transcribe() lazily loads+caches the model on first call rather
# than exposing a separate load-once object (unlike mlx_lm.load()). Warm it
# up here against silence so the model download/compile happens at startup,
# not on the first real utterance.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _warmup_audio() -> np.ndarray:
    """Real speech (the voice reference clip) exercises the whole decode path;
    a second of silence, the old warm-up, leaves the first real utterance to
    pay for compilation and decode-loop start-up."""
    try:
        with wave.open(os.path.join(ROOT, "assets", "reference_voice_short.wav")) as w:
            raw = w.readframes(min(w.getnframes(), w.getframerate() * 6))
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if w.getnchannels() > 1:
                x = x.reshape(-1, w.getnchannels()).mean(axis=1)
            return _resample_linear(x, w.getframerate(), 16000)
    except Exception:
        return np.zeros(16000, dtype=np.float32)


# Words Whisper otherwise gets wrong ("Skye" -> "Kai"/"sky"). It is only a
# hint about spelling and topic, not something it will say by itself.
VOCAB_PROMPT = "Skye. Einstein mode. Notion, Google Calendar, to-do list, expenses, budget, weather, alarm, reminder."
LOW_CONFIDENCE = -1.5
QUIET_RMS = 0.04
HALLUCINATIONS = {"thank you", "thanks", "thanks for watching", "thank you for watching", "you", "bye", "okay", "so", ""}
_recent = deque(maxlen=2)   # what he said last: makes follow-ups consistent
_log_path = None


def set_log_path(path: str):
    global _log_path
    _log_path = path


def remember(text: str):
    """Feed back a committed utterance so the next transcription has context."""
    t = (text or "").strip()
    if t:
        _recent.append(t[:160])


def _prompt() -> str:
    return (VOCAB_PROMPT + " " + " ".join(_recent)).strip()


def _resample_linear(samples: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    if orig_rate == target_rate:
        return samples
    duration = len(samples) / orig_rate
    target_len = int(round(duration * target_rate))
    orig_x = np.linspace(0, duration, num=len(samples), endpoint=False)
    target_x = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(target_x, orig_x, samples).astype(np.float32)


def transcribe_pcm(pcm_bytes: bytes, sample_rate: int = 16000, dtype: str = "int16") -> str:
    """Transcribe raw little-endian PCM audio (mono) to text.

    dtype: "int16" (from an Int16Array capture) or "float32".
    """
    if dtype == "int16":
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    else:
        samples = np.frombuffer(pcm_bytes, dtype="<f4")

    samples = _resample_linear(samples, sample_rate, 16000)

    t0 = time.time()
    with STT_LOCK:
        result = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=WHISPER_MODEL_REPO,
            language="en",
            fp16=True,
            initial_prompt=_prompt(),
            condition_on_previous_text=False,
        )
    text = result.get("text", "").strip()
    segs = result.get("segments") or []
    no_speech = max((g.get("no_speech_prob", 0) for g in segs), default=0)
    logprob = sum(g.get("avg_logprob", 0) for g in segs) / len(segs) if segs else 0
    dropped = None
    # Whisper's own "this was not speech" test, applied across the whole clip:
    # clicks, coughs and fan noise otherwise come back as "Ooh." or a stray
    # "Thank you.".
    if segs and no_speech > 0.6 and logprob < -1.0:
        dropped = "not speech"
    elif segs and max(g.get("compression_ratio", 0) for g in segs) > 2.4:
        dropped = "repetitive"          # "rotrotrotrot..." loops
    elif text and sum(ord(c) > 0x24F for c in text) / len(text) > 0.15:
        dropped = "wrong script"        # English-only: Korean/Arabic/Chinese output is a hallucination
    elif segs and logprob < LOW_CONFIDENCE:
        # Whisper invents fluent nonsense from clicks and hiss; measured on
        # synthetic noise its average log-probability was -7 to -9, against
        # about -0.2 for real speech.
        dropped = "low confidence"
    elif text.lower().strip(" .!") in HALLUCINATIONS and float(np.sqrt(np.mean(samples ** 2))) < QUIET_RMS:
        # Stock phrases Whisper emits for near-silence ("Thank you.", "you")
        # are only believed when the audio was actually loud enough to be speech.
        dropped = "stock phrase on quiet audio"
    elif text.lower().strip(" .") in VOCAB_PROMPT.lower() and float(np.sqrt(np.mean(samples ** 2))) < QUIET_RMS:
        # Same reasoning as the stock-phrase check above: Whisper parroting a
        # word straight out of its own prompt hint is only a hallucination if
        # the audio was too quiet to be real speech. Without the RMS check
        # this dropped every genuine, correctly-recognised "Skye" (the wake
        # word IS the first word of VOCAB_PROMPT) and one-word command like
        # "weather" or "budget" as if SKYE had imagined hearing them.
        dropped = "echoed the prompt"
    _log(len(samples) / 16000, text, no_speech, logprob, dropped, time.time() - t0)
    return "" if dropped else text


def _log(dur, text, no_speech, logprob, dropped, took):
    """One line per utterance: what Whisper heard, how sure it was, and whether it was thrown away."""
    if not _log_path:
        return
    try:
        with open(_log_path, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 1), "audio_s": round(dur, 2), "took_s": round(took, 2),
                                "text": text, "no_speech": round(float(no_speech), 2),
                                "avg_logprob": round(float(logprob), 2), "dropped": dropped}) + "\n")
    except OSError:
        pass


mlx_whisper.transcribe(_warmup_audio(), path_or_hf_repo=WHISPER_MODEL_REPO, language="en", initial_prompt=VOCAB_PROMPT)
print("✓ Whisper ONLINE")
