"""Local speech-to-text via mlx-whisper.

Runs entirely on-device (Metal via MLX) so voice input never leaves the
machine, unlike the browser's SpeechRecognition (which streams raw mic
audio to Apple/Google's cloud even though the rest of SKYE is offline).
"""

import threading

import numpy as np
import mlx_whisper

WHISPER_MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
STT_LOCK = threading.Lock()

print("Loading Whisper (mlx-whisper)...")
# mlx_whisper.transcribe() lazily loads+caches the model on first call rather
# than exposing a separate load-once object (unlike mlx_lm.load()). Warm it
# up here against silence so the model download/compile happens at startup,
# not on the first real utterance.
_WARMUP = np.zeros(16000, dtype=np.float32)
mlx_whisper.transcribe(_WARMUP, path_or_hf_repo=WHISPER_MODEL_REPO, language="en")
print("✓ Whisper ONLINE")


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

    with STT_LOCK:
        result = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=WHISPER_MODEL_REPO,
            language="en",
            fp16=True,
        )
    return result.get("text", "").strip()
