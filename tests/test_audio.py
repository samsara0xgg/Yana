from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import numpy as np

from typeless_local.audio import MicrophoneRecorder, VoiceActivityAnalyzer


def test_microphone_recorder_collects_audio_and_reports_level(monkeypatch) -> None:
    levels = []
    streams = []

    class FakeInputStream:
        def __init__(self, samplerate, channels, dtype, blocksize, callback, device=None):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.blocksize = blocksize
            self.callback = callback
            self.closed = False
            streams.append(self)

        def start(self) -> None:
            data = np.full((self.blocksize, self.channels), 0.05, dtype=np.float32)
            self.callback(data, self.blocksize, None, None)

        def stop(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=FakeInputStream),
    )
    recorder = MicrophoneRecorder(
        sample_rate=16000,
        channels=1,
        block_duration=0.05,
        on_level=levels.append,
    )

    recorder.start()
    audio = recorder.stop()

    assert streams[0].samplerate == 16000
    assert streams[0].blocksize == 800
    assert streams[0].closed is True
    assert np.allclose(audio, np.full(800, 0.05, dtype=np.float32))
    assert 0.0 < levels[0] <= 1.0


def test_microphone_recorder_closes_stream_when_stop_fails(monkeypatch) -> None:
    streams = []

    class FakeInputStream:
        def __init__(self, samplerate, channels, dtype, blocksize, callback, device=None):
            self.channels = channels
            self.blocksize = blocksize
            self.callback = callback
            self.aborted = False
            self.closed = False
            streams.append(self)

        def start(self) -> None:
            data = np.full((self.blocksize, self.channels), 0.05, dtype=np.float32)
            self.callback(data, self.blocksize, None, None)

        def stop(self) -> None:
            raise RuntimeError("stop failed")

        def abort(self) -> None:
            self.aborted = True

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=FakeInputStream),
    )
    recorder = MicrophoneRecorder(sample_rate=16000, channels=1, block_duration=0.05)

    recorder.start()
    audio = recorder.stop()

    assert streams[0].aborted is True
    assert streams[0].closed is True
    assert np.allclose(audio, np.full(800, 0.05, dtype=np.float32))


def test_a_stream_that_never_stops_does_not_hang_the_dictation(monkeypatch) -> None:
    """CoreAudio deadlocked inside stop() once and froze the whole app; now the audio is kept and it moves on."""

    stuck = threading.Event()

    class StuckInputStream:
        def __init__(self, samplerate, channels, dtype, blocksize, callback, device=None):
            self.channels, self.blocksize, self.callback = channels, blocksize, callback

        def start(self) -> None:
            self.callback(np.full((self.blocksize, self.channels), 0.05, dtype=np.float32), self.blocksize, None, None)

        def stop(self) -> None:
            stuck.wait()  # never returns while the test runs

        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=StuckInputStream))
    monkeypatch.setattr("typeless_local.audio.STOP_WAIT_S", 0.05)
    recorder = MicrophoneRecorder(sample_rate=16000, channels=1, block_duration=0.05)

    recorder.start()
    audio = recorder.stop()
    stuck.set()

    assert np.allclose(audio, np.full(800, 0.05, dtype=np.float32))
    assert not recorder.is_recording
    assert recorder.wedged, "the next start must not touch PortAudio again"


def test_voice_activity_analyzer_prefers_vocal_band_over_silence() -> None:
    analyzer = VoiceActivityAnalyzer(sample_rate=16000)
    silence = np.zeros(800, dtype=np.float32)
    t = np.arange(800, dtype=np.float32) / 16000
    voice = (0.12 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    silent_level = analyzer.analyze(silence)
    voice_levels = [analyzer.analyze(voice) for _ in range(6)]

    assert silent_level == 0.0
    assert voice_levels[0] > 0.05
    assert voice_levels[-1] == voice_levels[0]


def test_voice_activity_analyzer_responds_to_soft_speech_onset() -> None:
    """Soft initial syllables must produce a visible waveform level.

    Previously rms*0.35 ≈ voice_band_level ≈ noise_threshold for soft onsets,
    so the overlay stayed flat for the first ~half-second of dictation while
    audio was already being captured. Thresholds must be low enough that a
    quiet 440Hz tone (amplitude 0.025) registers above zero.
    """

    analyzer = VoiceActivityAnalyzer(sample_rate=16000)
    t = np.arange(800, dtype=np.float32) / 16000
    soft_voice = (0.025 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    level = analyzer.analyze(soft_voice)

    assert level > 0.0


def test_microphone_recorder_quality_gate_rejects_silence_and_short_audio() -> None:
    recorder = MicrophoneRecorder(sample_rate=16000)

    assert recorder.is_quality_ok(
        np.zeros(16000, dtype=np.float32),
        min_duration=0.25,
        low_volume_threshold=0.02,
    )[0] is False
    assert recorder.is_quality_ok(
        np.full(1600, 0.1, dtype=np.float32),
        min_duration=0.25,
        low_volume_threshold=0.02,
    )[0] is False
    assert recorder.is_quality_ok(
        np.full(16000, 0.1, dtype=np.float32),
        min_duration=0.25,
        low_volume_threshold=0.02,
    )[0] is True


def test_quality_gate_keeps_short_utterance_surrounded_by_silence() -> None:
    """A brief phrase must survive the volume gate.

    Whole-clip RMS averages speech into the silence around it, so short
    dictations scored below low_volume_threshold and were dropped before ASR
    with no output at all. Real trace.db drops on 2026-09-12 sat at RMS
    0.0174-0.0197 against a 0.02 gate while the speech itself was plainly
    audible. The gate measures the loudest 0.2s window instead.
    """

    recorder = MicrophoneRecorder(sample_rate=16000)
    t = np.arange(4800, dtype=np.float32) / 16000
    speech = (0.045 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    clip = np.concatenate(
        [np.zeros(6400, dtype=np.float32), speech, np.zeros(4800, dtype=np.float32)]
    )

    assert recorder.get_volume_level(clip) < 0.02
    assert recorder.peak_window_level(clip) > 0.02
    assert recorder.is_quality_ok(clip, min_duration=0.15, low_volume_threshold=0.02)[0] is True

    near_silence = (0.002 * np.sin(2 * np.pi * 220 * np.arange(16000, dtype=np.float32) / 16000)).astype(np.float32)
    assert recorder.is_quality_ok(near_silence, min_duration=0.15, low_volume_threshold=0.02)[0] is False


def test_microphone_recorder_can_retry_after_a_stream_fails_to_start(monkeypatch) -> None:
    streams = []

    class FakeInputStream:
        fail = True

        def __init__(self, samplerate, channels, dtype, blocksize, callback, device=None):
            self.closed = False
            self.started = False
            streams.append(self)

        def start(self) -> None:
            if FakeInputStream.fail:
                raise RuntimeError("Internal PortAudio error")
            self.started = True

        def stop(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=FakeInputStream))
    recorder = MicrophoneRecorder(sample_rate=16000)

    try:
        recorder.start()
    except RuntimeError:
        pass
    FakeInputStream.fail = False
    recorder.start()

    assert streams[0].closed is True
    assert len(streams) == 2 and streams[1].started is True
    assert recorder.is_recording


def test_a_stretch_ending_in_half_a_second_of_quiet_is_handed_on_while_recording() -> None:
    stretches = []
    recorder = MicrophoneRecorder(sample_rate=16000, on_stretch=stretches.append)
    speech = np.full(800, 0.05, dtype=np.float32)  # 50 ms blocks, as the stream delivers them
    quiet = np.zeros(800, dtype=np.float32)

    def feed(block, count):
        for _ in range(count):
            recorder._chunks.append(block)
            recorder._track_stretch(block)

    feed(speech, 60)  # 3 s of talk, then a pause: too short to cut
    feed(quiet, 12)
    assert stretches == []
    feed(speech, 50)  # 2.5 s more puts it past 5 s; the next half second of quiet cuts it
    feed(quiet, 9)
    assert stretches == []
    feed(quiet, 1)
    (first,) = stretches
    assert first.size == (60 + 12 + 50 + 10) * 800
    assert recorder.heard_until == first.size
    feed(quiet, 40)  # a long pause with no new words is not a stretch
    feed(speech, 10)
    assert len(stretches) == 1


def test_recorder_keeps_the_configured_channel_within_what_the_device_has(monkeypatch) -> None:
    opened = []

    class FakeInputStream:
        def __init__(self, samplerate, channels, dtype, blocksize, callback, device=None):
            self.channels, self.blocksize, self.callback = channels, blocksize, callback
            opened.append(channels)

        def start(self) -> None:
            data = np.tile(np.arange(self.channels, dtype=np.float32) * 0.1, (self.blocksize, 1))
            self.callback(data, self.blocksize, None, None)

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    inputs = {"max_input_channels": 2}
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=FakeInputStream, query_devices=lambda device, kind: inputs),
    )
    recorder = MicrophoneRecorder(input_channel=1)
    recorder.start()
    assert opened == [2] and np.allclose(recorder.stop(), 0.1)

    inputs["max_input_channels"] = 1
    recorder.start()
    assert opened == [2, 1] and np.allclose(recorder.stop(), 0.0)


def test_a_microphone_that_never_opens_fails_the_start_instead_of_hanging(monkeypatch) -> None:
    stuck = threading.Event()

    class HangingInputStream:
        def __init__(self, **kwargs) -> None:
            stuck.wait()  # CoreAudio never answers while the test runs

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=HangingInputStream))
    monkeypatch.setattr("typeless_local.audio.START_WAIT_S", 0.05)
    recorder = MicrophoneRecorder(sample_rate=16000, channels=1, block_duration=0.05)

    try:
        recorder.start()
    except RuntimeError as exc:
        assert "did not open" in str(exc)
    else:
        raise AssertionError("start() should have given up")
    assert recorder.wedged
    try:
        recorder.start()  # fails at once, without another wait on CoreAudio
    except RuntimeError as exc:
        assert "hung earlier" in str(exc)
    else:
        raise AssertionError("a wedged recorder should refuse to start")
    stuck.set()
    assert not recorder.is_recording


def test_waiting_for_the_press_moment_counts_the_devices_input_delay() -> None:
    import time

    recorder = MicrophoneRecorder(sample_rate=16000)
    recorder._stream = object()  # recording
    # A block whose first sample was captured 0.2 s before it reached the callback.
    late = SimpleNamespace(currentTime=100.2, inputBufferAdcTime=100.0)
    recorder._note_heard(800, late)
    arrived = time.monotonic()

    # Spoken 0.15 s ago: this block (spoken 0.2 s ago, 0.05 s long) does not yet cover it.
    assert recorder.wait_until_heard(arrived - 0.1, timeout=0.05) is False
    assert recorder.wait_until_heard(arrived - 0.2, timeout=0.05) is True

    feeder = threading.Timer(0.05, recorder._note_heard, args=(800, SimpleNamespace(currentTime=0, inputBufferAdcTime=0)))
    feeder.start()
    assert recorder.wait_until_heard(arrived, timeout=1.0) is True


def test_waiting_for_the_press_moment_ends_when_the_microphone_stops() -> None:
    recorder = MicrophoneRecorder(sample_rate=16000)
    recorder._stream = None

    assert recorder.wait_until_heard(1e12, timeout=1.0) is False
