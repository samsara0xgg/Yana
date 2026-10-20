"""Microphone recording primitives."""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
import wave
from typing import Callable

import numpy as np

LOGGER = logging.getLogger(__name__)
LevelCallback = Callable[[float], None]
StretchCallback = Callable[[np.ndarray], None]

# A stretch of at least 5 s that ends in 0.5 s of quiet is handed on while he
# goes on talking, so the stop leaves only the rest to hear. Quiet = a block
# under -45 dBFS; measured on 60 recordings, -40 dB cut into soft speech.
QUIET_RMS = 10 ** (-45 / 20)
PAUSE_SECONDS = 0.5
MIN_STRETCH_SECONDS = 5.0
# CoreAudio can deadlock stopping a stream (its IO thread and the stopping
# thread each wait on the other's lock, seen when the audio device changes).
# The stop gets this long, then the dictation carries on without it.
STOP_WAIT_S = 2.0
# Opening the stream can hang in CoreAudio the same way; past this the start counts as failed.
START_WAIT_S = 4.0


def peak_level(audio: np.ndarray, sample_rate: int, window_seconds: float = 0.2) -> float:
    """RMS of the loudest short window.

    Whole-clip RMS averages a brief phrase into the silence around it, so a
    short utterance scores lower than the same speech in a long recording
    and gets dropped as "too quiet". The loudest window is independent of
    how much silence surrounds it.
    """

    normalized = np.asarray(audio, dtype=np.float32)
    window = max(1, int(sample_rate * window_seconds))
    if normalized.size <= window:
        if normalized.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(normalized), dtype=np.float64)))
    cumulative = np.concatenate(([0.0], np.cumsum(np.square(normalized, dtype=np.float64))))
    window_means = (cumulative[window:] - cumulative[:-window]) / window
    return float(np.sqrt(window_means.max()))


def keep_recording(
    folder: Path, name: str, audio: np.ndarray, sample_rate: int, keep: int
) -> Path:
    """Write ``audio`` to ``folder/name.wav`` and delete all but the newest ``keep``.

    Names must sort in time order, since pruning goes by name.
    """

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.wav"
    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((samples * 32767).astype("<i2").tobytes())
    for old in sorted(folder.glob("*.wav"))[:-keep]:
        old.unlink()
    return path


class VoiceActivityAnalyzer:
    """Convert microphone chunks into a smoothed voice-band activity level."""

    def __init__(
        self,
        sample_rate: int,
        fft_size: int = 2048,
        noise_threshold: float = 0.01,
        voice_threshold: float = 0.06,
    ) -> None:
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.noise_threshold = noise_threshold
        self.voice_threshold = voice_threshold
        self._smoothed = 0.0

    def analyze(self, chunk: np.ndarray) -> float:
        """Return a 0..1 level based mostly on vocal-frequency energy."""

        if chunk.size == 0:
            self._smoothed *= 0.25
            return self._smoothed

        raw = max(self._rms_level(chunk) * 0.35, self._voice_band_level(chunk))
        if raw <= self.noise_threshold:
            target = 0.0
        elif raw >= self.voice_threshold:
            target = min(1.0, (raw - self.noise_threshold) / (self.voice_threshold * 2.8))
        else:
            target = ((raw - self.noise_threshold) / (self.voice_threshold - self.noise_threshold)) * 0.35

        if target >= self._smoothed:
            self._smoothed = target
        else:
            self._smoothed = self._smoothed * 0.25 + target * 0.75
        return max(0.0, min(1.0, self._smoothed))

    def _rms_level(self, chunk: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
        return max(0.0, min(1.0, rms * 8.0))

    def _voice_band_level(self, chunk: np.ndarray) -> float:
        fft_size = self.fft_size
        padded = np.zeros(fft_size, dtype=np.float32)
        sample_count = min(chunk.size, fft_size)
        window = np.hanning(sample_count).astype(np.float32)
        padded[:sample_count] = chunk[:sample_count] * window

        spectrum = np.abs(np.fft.rfft(padded)) / (fft_size / 2.0)
        start = min(14, spectrum.size - 1)
        end = min(140, spectrum.size)
        if end <= start:
            return 0.0
        band = spectrum[start:end]
        band_rms = float(np.sqrt(np.mean(np.square(band), dtype=np.float64)))
        return max(0.0, min(1.0, band_rms * 22.0))


class MicrophoneRecorder:
    """Record mono float32 microphone audio until stopped."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_duration: float = 0.05,
        on_level: LevelCallback | None = None,
        device: int | None = None,
        on_stretch: StretchCallback | None = None,
        input_channel: int = 0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        # Which of the device's channels to keep. The reSpeaker XVF3800 sends its
        # hot, clipping AGC mix on 0 and a cleaner beam on 1.
        self.input_channel = input_channel
        self.block_duration = block_duration
        self.on_level = on_level
        # Reassigned when the menu picks another input; None means system default.
        self.device = device
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None
        # Set once CoreAudio has hung opening or closing a stream. Touching
        # PortAudio again (another open, a device refresh) can hang the caller
        # for good, so nothing more is tried until the app restarts.
        self.wedged = False
        self._started_at = 0.0
        self._analyzer = VoiceActivityAnalyzer(sample_rate)
        # Called on the audio thread with each stretch cut at a pause; keep it quick.
        self.on_stretch = on_stretch
        # Called on the audio thread with every block, for recognition that streams; keep it quick.
        self.on_chunk: StretchCallback | None = None
        self.heard_until = 0  # samples already handed to on_stretch this recording
        self._stretch_from = 0
        self._stretch_samples = 0
        self._quiet_samples = 0
        self._spoke = False

    @property
    def is_recording(self) -> bool:
        """Whether an input stream is currently open."""

        return self._stream is not None

    @property
    def elapsed(self) -> float:
        """Elapsed recording seconds."""

        if not self._started_at:
            return 0.0
        return time.monotonic() - self._started_at

    @property
    def chunk_count(self) -> int:
        """Number of audio chunks received for the active recording."""

        with self._lock:
            return len(self._chunks)

    def get_volume_level(self, audio: np.ndarray) -> float:
        """Compute normalized RMS volume for mono audio."""

        normalized = np.asarray(audio, dtype=np.float32)
        if normalized.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(normalized), dtype=np.float64)))

    def peak_window_level(self, audio: np.ndarray, window_seconds: float = 0.2) -> float:
        """RMS of the loudest short window (see :func:`peak_level`)."""

        return peak_level(audio, self.sample_rate, window_seconds)

    def _track_stretch(self, chunk: np.ndarray) -> None:
        """Per block: after 0.5 s of quiet, hand the stretch before it to ``on_stretch``."""

        if self.on_stretch is None or chunk.size == 0:
            return
        quiet = self.get_volume_level(chunk) < QUIET_RMS
        self._quiet_samples = self._quiet_samples + chunk.size if quiet else 0
        self._spoke = self._spoke or not quiet
        self._stretch_samples += chunk.size
        if not self._spoke or self._quiet_samples < PAUSE_SECONDS * self.sample_rate:
            return
        if self._stretch_samples < MIN_STRETCH_SECONDS * self.sample_rate:
            return
        with self._lock:
            stretch = np.concatenate(self._chunks[self._stretch_from :])
            self._stretch_from = len(self._chunks)
        self.heard_until += stretch.size
        self._stretch_samples, self._spoke = 0, False
        self.on_stretch(stretch)

    def is_quality_ok(
        self,
        audio: np.ndarray,
        *,
        min_duration: float,
        low_volume_threshold: float,
    ) -> tuple[bool, str]:
        """Validate audio before ASR to avoid Whisper silence hallucinations."""

        normalized = np.asarray(audio, dtype=np.float32)
        duration_seconds = normalized.size / self.sample_rate
        volume_level = self.peak_window_level(normalized)
        issues: list[str] = []

        if normalized.size == 0:
            issues.append("audio is empty")
        if duration_seconds < min_duration:
            issues.append(
                f"duration {duration_seconds:.2f}s is below minimum {min_duration:.2f}s"
            )
        if volume_level < low_volume_threshold:
            issues.append(
                "peak volume "
                f"{volume_level:.4f} is below threshold {low_volume_threshold:.4f}"
            )

        if issues:
            return False, f"Audio quality warning: {'; '.join(issues)}."
        return True, "Audio quality check passed."

    def start(self) -> None:
        """Start recording."""

        if self._stream is not None:
            return
        if self.wedged:
            raise MicrophoneWedged("the microphone hung earlier; restart the app")

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("sounddevice is required for microphone recording.") from exc

        self._chunks = []
        self._started_at = time.monotonic()
        self.heard_until = self._stretch_from = self._stretch_samples = self._quiet_samples = 0
        self._spoke = False
        blocksize = max(1, int(self.sample_rate * self.block_duration))
        channel, channels = self.input_channel, self.channels
        if channel:
            available = int(sd.query_devices(self.device, "input")["max_input_channels"])
            channel = min(channel, available - 1)
            channels = max(channels, channel + 1)

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                LOGGER.warning("Audio input status: %s", status)
            chunk = np.asarray(indata[:, channel], dtype=np.float32).copy()
            with self._lock:
                self._chunks.append(chunk)
            on_chunk = self.on_chunk
            if on_chunk is not None:
                on_chunk(chunk)
            self._track_stretch(chunk)
            if self.on_level is not None and chunk.size:
                self.on_level(self._analyzer.analyze(chunk))

        opened: dict = {}
        done = threading.Event()
        handoff = threading.Lock()  # so a stream is either taken here or released there, never neither

        def open_stream() -> None:
            try:
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=blocksize,
                    callback=callback,
                    device=self.device,
                )
                try:
                    stream.start()
                except Exception:
                    # Left assigned, a stream that never started would make the next
                    # start() return early and "record" nothing.
                    stream.close()
                    raise
                opened["stream"] = stream
            except Exception as exc:
                opened["error"] = exc
            finally:
                with handoff:
                    done.set()
                    late = opened.get("abandoned")
                if late and "stream" in opened:
                    _release(opened["stream"], threading.Event())  # it came up after all, too late

        # Opened off the calling thread (the hotkey's): CoreAudio has hung here before.
        threading.Thread(target=open_stream, name="mic-start", daemon=True).start()
        done.wait(START_WAIT_S)
        with handoff:
            if not done.is_set():
                opened["abandoned"] = True
                self.wedged = True
                raise MicrophoneWedged(f"the microphone did not open within {START_WAIT_S:.0f}s")
        if "error" in opened:
            raise opened["error"]
        self._stream = opened["stream"]
        LOGGER.info("Microphone recording started")

    def stop(self) -> np.ndarray:
        """Stop recording and return captured mono audio."""

        stream = self._stream
        self._stream = None
        if stream is not None:
            released = threading.Event()
            threading.Thread(target=_release, args=(stream, released), name="mic-stop", daemon=True).start()
            if not released.wait(STOP_WAIT_S):
                # ponytail: the stuck stream and its thread are abandoned until the app quits.
                self.wedged = True
                LOGGER.warning("Microphone stream did not stop within %.1fs; keeping the audio and moving on", STOP_WAIT_S)
        with self._lock:
            chunks = list(self._chunks)
            self._chunks = []
        self._started_at = 0.0
        if not chunks:
            return np.empty(0, dtype=np.float32)
        audio = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        LOGGER.info("Microphone recording stopped with %.2fs audio", audio.size / self.sample_rate)
        return audio


class MicrophoneWedged(RuntimeError):
    """CoreAudio hung on this process's microphone; only a restart clears it."""


def _release(stream, released: threading.Event) -> None:
    """Stop and close a stream; on its own thread, because CoreAudio may never return."""

    try:
        stream.stop()
    except Exception:
        LOGGER.warning("Failed to stop microphone stream cleanly", exc_info=True)
        abort = getattr(stream, "abort", None)
        if abort is not None:
            try:
                abort()
            except Exception:
                LOGGER.debug("Failed to abort microphone stream", exc_info=True)
    finally:
        try:
            stream.close()
        except Exception:
            LOGGER.warning("Failed to close microphone stream", exc_info=True)
        released.set()
