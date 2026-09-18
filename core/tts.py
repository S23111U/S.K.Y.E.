"""Local text-to-speech via XTTS-v2 (coqui-tts), voice-cloned from a reference clip.

Two compatibility fixes are required on this stack (torch 2.9 + transformers 5.x,
needed for the Gemma 4 LLM) that coqui-tts's code doesn't natively support:

1. `torchaudio.load` now requires torchcodec, which requires ffmpeg (not
   installed on this machine, and deliberately avoided elsewhere in this
   project). Since the only thing coqui-tts loads via torchaudio is our own
   plain WAV reference clip, it's replaced with a soundfile-based loader.
2. `transformers.pytorch_utils.isin_mps_friendly` was removed in transformers
   5.x; it was only ever a torch<=2.3 MPS workaround, so `torch.isin` is a
   correct direct substitute on this project's torch version.

Note: XTTS's own `inference_stream()` (sub-sentence streaming) is NOT used
here — its vendored streaming generator calls a private transformers
GenerationMixin method with a signature removed in transformers 5.x, and
patching that open a much deeper compatibility gap (a KV-cache mismatch that
causes unbounded MPS memory growth, not just an AttributeError). The
standard (non-streaming) `model.inference()` per sentence is used instead;
it uses transformers' actively-maintained generate() path and is unaffected.
Playback still starts before the full reply is synthesized, one sentence at
a time, matching what the old browser.html speak() did.
"""

import os
import random
import re
import threading

import numpy as np
import soundfile as sf
import torch


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _load_audio_no_ffmpeg(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    tensor = torch.from_numpy(data.T.copy())
    return tensor, sr


import torchaudio  # noqa: E402

torchaudio.load = _load_audio_no_ffmpeg

import transformers.pytorch_utils as _ptu  # noqa: E402

if not hasattr(_ptu, "isin_mps_friendly"):
    _ptu.isin_mps_friendly = torch.isin

from TTS.tts.configs.xtts_config import XttsConfig  # noqa: E402
from TTS.tts.models.xtts import Xtts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_VOICE = os.path.join(ROOT, "assets", "reference_voice.wav")
XTTS_CHECKPOINT_DIR = os.path.expanduser(
    "~/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2"
)

TTS_LOCK = threading.Lock()
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

if not os.path.isfile(REFERENCE_VOICE):
    raise RuntimeError(
        f"Missing {REFERENCE_VOICE} — place a 6-30s mono WAV of a single "
        "speaker there (your own recording, or a public-domain/licensed "
        "clip) before starting TTS. Never a real identifiable person's "
        "actual voice without their consent for cloning."
    )

print("Loading XTTS-v2...")
with TTS_LOCK:
    _config = XttsConfig()
    _config.load_json(os.path.join(XTTS_CHECKPOINT_DIR, "config.json"))
    _model = Xtts.init_from_config(_config)
    _model.load_checkpoint(_config, checkpoint_dir=XTTS_CHECKPOINT_DIR)
    _model.to(_DEVICE)
    _gpt_cond_latent, _speaker_embedding = _model.get_conditioning_latents(
        audio_path=[REFERENCE_VOICE]
    )
print(f"✓ XTTS-v2 ONLINE ({_DEVICE})")


def synthesize_reply(
    text: str,
    language: str = "en",
    *,
    temperature_range: tuple[float, float] = (0.75, 0.95),
    speed_range: tuple[float, float] = (0.95, 1.08),
):
    """Yields (pcm_float32_bytes, is_final) for each sentence in `text`, at 24kHz mono.

    Synthesizing one sentence at a time (rather than the whole reply) lets
    playback start before the full reply is ready, without depending on
    XTTS's broken-on-this-stack sub-sentence streaming API.

    `temperature`/`speed` are jittered per sentence (within a tasteful range)
    rather than held at one fixed value. A single fixed setting produces the
    same cadence and inflection on every sentence, which reads as flat and
    mechanical over a whole reply; small random per-sentence variation is
    what a real speaker's natural pacing looks like.
    """
    sentences = split_sentences(text) or [text]
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return

    with TTS_LOCK:
        for i, sentence in enumerate(sentences):
            out = _model.inference(
                sentence,
                language,
                _gpt_cond_latent,
                _speaker_embedding,
                temperature=random.uniform(*temperature_range),
                repetition_penalty=2.0,
                speed=random.uniform(*speed_range),
            )
            pcm = np.asarray(out["wav"], dtype=np.float32).tobytes()
            yield pcm, i == len(sentences) - 1
