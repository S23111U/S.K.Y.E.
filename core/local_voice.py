"""Local microphone capture and speaker playback for the CLI's voice mode.

The browser UI does this in JS against a WebRTC MediaStream (with real
echo-cancellation from the browser's audio pipeline). This is the CLI's
equivalent, direct against the Mac's audio hardware via PyAudio, so `:voice`
works without ever opening a browser tab.

Two things are deliberately simpler than the browser path, both to avoid
needing acoustic echo cancellation (which PyAudio does not give us):

  - No wake word. `:voice` is an explicit, one-time opt-in per session — the
    user turns it on when they want hands-free, and off when they don't — so
    there is no "always listening for 'Skye'" mic session to worry about
    (see the same reasoning behind the browser's keypress-to-wake change).
  - No voice barge-in. The mic is paused while SKYE is talking (there is no
    AEC layer here to tell "the user talking" apart from "SKYE's own voice
    coming back through the speaker"); interrupting playback is by pressing
    any key instead, checked between audio chunks.
"""

import queue
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import pyaudio

MIC_RATE = 16000
SPEAKER_RATE = 24000
BLOCK = 512                      # ~32 ms at 16 kHz

# Same shape of constants as clients/browser.html's VAD (see the comments
# there for why these particular values), ported to Python.
ENERGY_FLOOR = 0.018
NOISE_MULT = 3.2
SILENCE_MS = 1000
MIN_UTTERANCE_MS = 400
MAX_UTTERANCE_MS = 15000
CALIBRATE_BLOCKS = 40             # ~1.3 s of room noise before first listen


class MicUnavailable(Exception):
    pass


def _rms(int16_block: np.ndarray) -> float:
    f = int16_block.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(f * f))) if len(f) else 0.0


class LocalMic:
    """One PyAudio input stream, opened lazily and reused across utterances."""

    def __init__(self):
        self._pa = None
        self._stream = None
        self._noise_floor = 0.006

    def _ensure_open(self):
        if self._stream is not None:
            return
        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paInt16, channels=1, rate=MIC_RATE,
                input=True, frames_per_buffer=BLOCK,
            )
        except Exception as e:
            self.close()
            raise MicUnavailable(str(e))

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def _calibrate(self):
        levels = []
        for _ in range(CALIBRATE_BLOCKS):
            data = np.frombuffer(self._stream.read(BLOCK, exception_on_overflow=False), dtype="<i2")
            levels.append(_rms(data))
        if levels:
            levels.sort()
            self._noise_floor = min(0.02, max(0.003, levels[len(levels) // 2]))

    def listen_utterance(self, stop_event: threading.Event | None = None) -> bytes | None:
        """Blocks until one complete utterance is captured, returns 16-bit PCM
        at 16 kHz (ready for stt.transcribe_pcm), or None if `stop_event` was
        set, or nothing speech-like happened before a silence timeout with an
        empty buffer."""
        self._ensure_open()
        self._calibrate()
        threshold = max(ENERGY_FLOOR, self._noise_floor * NOISE_MULT)
        chunks = []
        silence_start = None
        started = False
        block_ms = BLOCK / MIC_RATE * 1000
        elapsed = 0.0
        while True:
            if stop_event is not None and stop_event.is_set():
                return None
            data = self._stream.read(BLOCK, exception_on_overflow=False)
            arr = np.frombuffer(data, dtype="<i2")
            level = _rms(arr)
            is_speech = level > threshold
            if is_speech:
                chunks.append(arr)
                started = True
                silence_start = None
            elif started:
                chunks.append(arr)          # keep trailing silence for natural cadence
                if silence_start is None:
                    silence_start = elapsed
                elif elapsed - silence_start >= SILENCE_MS:
                    break
            elif not started and elapsed > 20_000:      # nobody has said anything in 20s
                return b""
            elapsed += block_ms
            if started and elapsed >= MAX_UTTERANCE_MS:
                break
        if not chunks:
            return b""
        pcm = np.concatenate(chunks)
        dur_ms = len(pcm) / MIC_RATE * 1000
        if dur_ms < MIN_UTTERANCE_MS:
            return b""       # a click/cough, not speech — same floor as the browser's VAD
        return pcm.astype("<i2").tobytes()


class LocalSpeaker:
    """Plays the float32 PCM chunks tts_client.synthesize_reply() yields."""

    def __init__(self):
        self._pa = None
        self._stream = None

    def _ensure_open(self):
        if self._stream is not None:
            return
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(format=pyaudio.paFloat32, channels=1, rate=SPEAKER_RATE, output=True)

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def write_chunk(self, pcm: bytes):
        """Writes one already-decoded float32 PCM chunk immediately (for
        streaming playback as chunks arrive over the wire, one at a time)."""
        self._ensure_open()
        if pcm:
            self._stream.write(pcm)

    def play(self, chunk_iter, stop_event: threading.Event | None = None):
        """Writes each (pcm_bytes, is_final) chunk to the output device,
        checking `stop_event` between chunks so a keypress can cut it off."""
        for pcm, _is_final in chunk_iter:
            if stop_event is not None and stop_event.is_set():
                return
            self.write_chunk(pcm)


def any_key_pressed(poll_s: float = 0.0) -> bool:
    """Non-blocking check for a pending keypress on stdin (Unix terminals
    only — this app only ships for macOS). Used so 'press any key to
    interrupt' works while audio is actively playing, without blocking on
    input(). The key itself is left in the input buffer if it's a newline
    the caller's next input() will want; a stray character before it is
    swallowed, which is an acceptable trade for a keypress-to-stop UX."""
    if not sys.stdin.isatty():
        return False
    r, _, _ = select.select([sys.stdin], [], [], poll_s)
    return bool(r)


def drain_stdin():
    """Consumes whatever keypress(es) triggered any_key_pressed(), so they
    don't leak into the next input() prompt as stray characters."""
    if not sys.stdin.isatty():
        return
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        pass


class BargeWatcher:
    """Background thread: while armed, sets `stop` the moment any key is
    pressed. Used during local TTS playback as the keypress equivalent of the
    browser's voice barge-in."""

    def __init__(self):
        self.stop = threading.Event()
        self._armed = threading.Event()
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._alive:
            if self._armed.is_set() and any_key_pressed(0.05):
                self.stop.set()
                self._armed.clear()
                drain_stdin()
            else:
                time.sleep(0.03)

    def arm(self):
        self.stop.clear()
        self._armed.set()

    def disarm(self):
        self._armed.clear()

    def shutdown(self):
        self._alive = False
