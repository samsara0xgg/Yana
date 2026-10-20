from __future__ import annotations

from collections import deque
import dataclasses

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pytest

from typeless_local import app as app_module
from typeless_local.app import Insertion, TypelessLocalApp, count_units
from typeless_local.asr import Transcript
from typeless_local.capsule import Capsule
from typeless_local.mac_integration import FocusContext
from typeless_local.menubar import Recent
from typeless_local.preferences import Preferences
from typeless_local.refine import MissingAPIKey, RefineResult

TARGET_PID = 4242


class _FakeOverlay:
    """Records what the capsule was asked to draw."""

    def __init__(self) -> None:
        self.calls = []

    def setup(self) -> None:
        self.calls.append(("setup",))

    def show(self, state, **data) -> None:
        self.calls.append(("show", state, data))

    def hide(self) -> None:
        self.calls.append(("hide",))

    def end_edit(self) -> None:
        self.calls.append(("end_edit",))

    def update_level(self, level) -> None:
        self.calls.append(("level", level))

    def set_handle(self, on) -> None:
        self.calls.append(("handle", on))

    def set_anchor(self, mode, caret=None) -> None:
        self.calls.append(("anchor", mode, caret))

    def shown(self) -> list[str]:
        return [call[1] for call in self.calls if call[0] == "show"]

    def last(self):
        return [call for call in self.calls if call[0] in ("show", "hide")][-1]


class _ManualTimer:
    """A threading.Timer that only fires when the test says so."""

    made: list["_ManualTimer"] = []

    def __init__(self, delay, callback) -> None:
        self.delay = delay
        self.callback = callback
        self.daemon = False
        self.cancelled = False
        _ManualTimer.made.append(self)

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback()


def _attach(app) -> _FakeOverlay:
    """Give an app built with __new__ a capsule that draws into a fake overlay."""

    overlay = _FakeOverlay()
    app.overlay = overlay
    app.capsule = Capsule(overlay, lambda fn, *a, **k: fn(*a, **k), timer=_ManualTimer)
    return overlay


@pytest.fixture(autouse=True)
def _mac(monkeypatch):
    """Keep the tests off the real Mac: permissions, the frontmost app, keys."""

    _ManualTimer.made = []
    monkeypatch.setattr(app_module.permissions, "microphone_status", lambda: "authorized")
    monkeypatch.setattr(app_module, "frontmost_pid", lambda: TARGET_PID)
    monkeypatch.setattr(app_module, "prepare_paste", lambda context: "blind")
    monkeypatch.setattr(app_module, "focused_text_length", lambda pid: None)
    monkeypatch.setattr(app_module, "caret_rect", lambda: None)
    monkeypatch.setattr(app_module, "undo_last_edit", lambda: None)


class _FakeASR:
    def __init__(self, text: str, language: str = "en", confidence: float = 0.9) -> None:
        self.text = text
        self.language = language
        self.confidence = confidence
        self.calls = []
        self.prompts = []
        self.languages = []
        self.probs = {}

    def detect_language(self, audio: np.ndarray) -> dict:
        return self.probs

    def transcribe(self, audio: np.ndarray, initial_prompt: str | None = None, language: str | None = None) -> Transcript:
        self.calls.append(audio)
        self.prompts.append(initial_prompt)
        self.languages.append(language)
        return Transcript(text=self.text, language=self.language, confidence=self.confidence)


class _FakeRefiner:
    def __init__(self) -> None:
        self.calls = []

    def refine(self, text: str, context: FocusContext, vocab=None, language="") -> RefineResult:
        self.calls.append((text, context))
        self.language = language
        return RefineResult(text="Refined text.", raw_text=text, model="gpt-5.4-mini")


class _FailingHotkeys:
    def start(self) -> None:
        raise RuntimeError("no accessibility")


class _FakeHotkeys:
    """Mimics GlobalHotkeyMonitor's event-time fields used by _on_primary_*."""

    def __init__(self) -> None:
        self.last_primary_down_at = 0.0
        self.last_primary_up_at = 0.0

    def start(self) -> None:
        return None


class _FakeRecorder:
    def __init__(self) -> None:
        self.stopped = False
        self.started = False
        self.fail_start = False
        self.fail_stop = False
        self.elapsed = 1.0
        self.chunk_count = 1

    def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("mic unavailable")
        self.started = True

    def stop(self) -> np.ndarray:
        if self.fail_stop:
            raise RuntimeError("mic stop failed")
        self.stopped = True
        return np.ones(16000, dtype=np.float32)

    def is_quality_ok(self, audio, *, min_duration, low_volume_threshold):
        duration_seconds = audio.size / 16000
        volume = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0.0
        if audio.size == 0 or duration_seconds < min_duration or volume < low_volume_threshold:
            return False, "low quality"
        return True, "ok"


class _FakeExecutor:
    def __init__(self) -> None:
        self.submissions = []

    def submit(self, fn, *args):
        self.submissions.append((fn, args))
        return SimpleNamespace(add_done_callback=lambda callback: None)


class _FakeDucker:
    def __init__(self) -> None:
        self.calls = []

    def duck(self) -> bool:
        self.calls.append("duck")
        return True

    def restore(self) -> None:
        self.calls.append("restore")

    def restore_all(self) -> None:
        self.calls.append("restore_all")


def _make_app(raw_text: str) -> TypelessLocalApp:
    app = TypelessLocalApp.__new__(TypelessLocalApp)
    app.config = SimpleNamespace(sample_rate=16000, min_recording_seconds=0.25, low_volume_threshold=0.02)
    _attach(app)
    app.asr = _FakeASR(raw_text)
    app.refiner = _FakeRefiner()
    app.recorder = _FakeRecorder()
    app._lock = threading.RLock()
    app.state = "processing"
    app._copy_fallback_text = ""
    return app


@pytest.fixture(autouse=True)
def _media_keys(monkeypatch) -> list[str]:
    """Never tap the real play/pause key or read the real mixer from a test."""

    presses: list[str] = []
    monkeypatch.setattr(app_module.devices, "playing_apps", lambda: set())
    monkeypatch.setattr(app_module.devices, "is_call_app", lambda pid: False)
    monkeypatch.setattr(app_module, "press_play_pause", lambda: presses.append("play_pause"))
    return presses


def _recording_app() -> TypelessLocalApp:
    app = TypelessLocalApp.__new__(TypelessLocalApp)
    app._lock = threading.RLock()
    app._recording_timer = None
    app.config = SimpleNamespace(sample_rate=16000, min_recording_seconds=0.25, max_recording_seconds=0)
    app.state = "idle"
    app.mode = "tap"
    _attach(app)
    app.recorder = _FakeRecorder()
    app.audio_ducker = _FakeDucker()
    app.executor = _FakeExecutor()
    app.focus_context = FocusContext(app_name="", window_title="")
    return app


def test_process_audio_transcribes_refines_and_pastes(monkeypatch) -> None:
    pasted = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("raw dictation")
    context = FocusContext(app_name="TextEdit", window_title="Untitled", can_insert_text=True)

    app._process_audio(np.ones(16000, dtype=np.float32), context)

    assert app.asr.calls
    assert app.refiner.calls == [("raw dictation", context)]
    assert pasted == ["Refined text."]
    assert app.overlay.shown() == ["refining", "inserted"]
    assert app.overlay.calls[-1] == ("show", "inserted", {"n": 2, "replaced": False})
    # Until the user types, the capsule can still take the paste back.
    assert app._insertion == Insertion("Refined text.", "raw dictation", TARGET_PID, context)
    assert app._keys_wanted is True
    assert app.state == "idle"


def test_the_guide_sees_what_was_heard_and_what_went_in(monkeypatch) -> None:
    monkeypatch.setattr("typeless_local.app.paste_text", lambda text: None)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("raw dictation")
    told = []
    app.windows = SimpleNamespace(guide=told.append)
    app._call_ui = lambda callback, *args: callback(*args)

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("TextEdit", "", can_insert_text=True))

    assert told == [{"t": "result", "raw": "raw dictation", "text": "Refined text."}]


def test_process_audio_joins_stretches_heard_while_talking_with_the_rest(monkeypatch) -> None:
    from concurrent.futures import Future

    monkeypatch.setattr("typeless_local.app.paste_text", lambda _text: None)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("the rest")
    heard = Future()
    heard.set_result(Transcript(text="第一段。", language="zh", confidence=0.9))
    audio = np.concatenate([np.full(16000, 0.5, dtype=np.float32), np.ones(8000, dtype=np.float32)])
    context = FocusContext(app_name="TextEdit", window_title="Untitled", can_insert_text=True)

    app._process_audio(audio, context, None, [heard], 16000)

    # Only the part after the cut goes to Whisper; the words join in order.
    (tail,) = app.asr.calls
    assert tail.size == 8000
    assert app.refiner.calls == [("第一段。the rest", context)]


def test_word_list_goes_to_whisper_only_for_chunks_within_one_window() -> None:
    app = _make_app("hi")
    app.whisper_prompt = "Common terms: Jarvis, StarTrial."

    app._hear(np.full(16000 * 10, 0.5, dtype=np.float32))
    app._hear(np.full(16000 * 31, 0.5, dtype=np.float32))

    assert app.asr.prompts == ["Common terms: Jarvis, StarTrial.", None]


def test_chinese_as_the_spoken_language_asks_whisper_for_simplified() -> None:
    """Only a forced-Chinese recognizer gets the Simplified hint: Automatic and
    English must not be pushed toward Chinese."""

    app = _make_app("hi")
    app.whisper_prompt = "Common terms: Jarvis."
    for spoken in ("zh", "", "en"):
        app.asr._asr_config = {"language": spoken}
        app._hear(np.full(16000 * 10, 0.5, dtype=np.float32))
    app.asr._asr_config = {}
    app.whisper_prompt = ""
    app._hear(np.full(16000 * 10, 0.5, dtype=np.float32))
    app.asr._asr_config = {"language": "zh"}
    app._hear(np.full(16000 * 31, 0.5, dtype=np.float32))

    assert app.asr.prompts == [
        f"{app_module.SIMPLIFIED_HINT} Common terms: Jarvis.",
        "Common terms: Jarvis.",
        "Common terms: Jarvis.",
        None,
        None,
    ]


def test_process_audio_hands_the_text_over_in_a_card_when_focus_is_not_editable(monkeypatch) -> None:
    """With nowhere to paste, the text still lands on the clipboard by itself.

    It used to wait for a click on the overlay's Copy button, so a dictation
    aimed at a non-editable target was one missed click away from being lost.
    """

    pasted = []
    copied = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", copied.append)
    app = _make_app("raw dictation")
    context = FocusContext(app_name="Finder", window_title="Desktop", focused_role="AXGroup")

    app._process_audio(np.ones(16000, dtype=np.float32), context)

    assert pasted == []
    assert copied == ["Refined text."]
    assert app._copy_fallback_text == "Refined text."
    assert app.overlay.calls[-1] == ("show", "edit-notarget", {"text": "Refined text."})
    assert getattr(app, "_insertion", None) is None


def test_copy_last_transcript_sets_the_clipboard_and_says_so(monkeypatch) -> None:
    copied = []
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", copied.append)
    app = _make_app("raw dictation")
    app._copy_fallback_text = "Refined text."

    app._copy_last_transcript()

    assert copied == ["Refined text."]
    assert app.overlay.calls[-1] == ("show", "copied", {})


def test_process_audio_empty_transcript_does_not_paste(monkeypatch) -> None:
    pasted = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    app = _make_app("")
    app._capture_device = "MacBook Pro 麦克风"

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("", ""))

    assert pasted == []
    assert app.overlay.calls[-1] == ("show", "empty", {"device": "MacBook Pro 麦克风"})
    assert app.state == "idle"


def test_process_audio_drops_low_quality_audio_before_asr(monkeypatch) -> None:
    pasted = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    app = _make_app("hallucinated prior")

    app._process_audio(np.zeros(16000, dtype=np.float32), FocusContext("", ""))

    assert app.asr.calls == []
    assert pasted == []
    assert app.overlay.shown() == ["empty"]


def test_process_audio_drops_short_non_zh_fragment(monkeypatch) -> None:
    pasted = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    app = _make_app("you")
    app.asr = _FakeASR("you", language="en", confidence=0.31)

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("", ""))

    assert app.asr.calls
    assert app.refiner.calls == []
    assert pasted == []
    assert app.overlay.shown() == ["empty"]


def test_empty_capsule_goes_away_by_itself() -> None:
    app = _make_app("")
    app._process_audio(np.zeros(16000, dtype=np.float32), FocusContext("", ""))

    (timer,) = _ManualTimer.made
    assert timer.delay == 2.4
    timer.fire()

    assert app.overlay.calls[-1] == ("hide",)


def test_hands_free_hotkey_upgrades_active_tap_recording() -> None:
    app = TypelessLocalApp.__new__(TypelessLocalApp)
    app._lock = threading.RLock()
    app.state = "recording"
    app.mode = "tap"
    app.config = SimpleNamespace(max_recording_seconds=540)
    overlay = _attach(app)

    app._on_hotkey("hands_free")

    assert app.mode == "hands_free"
    ((kind, state, data),) = overlay.calls
    assert (kind, state, data["mode"], data["max"]) == ("show", "rec", "latch", 540.0)


def test_start_shows_permission_state_when_hotkey_install_fails(monkeypatch) -> None:
    monkeypatch.setattr("typeless_local.app.has_accessibility_trust", lambda: False)
    monkeypatch.setattr("typeless_local.app.request_accessibility_trust", lambda: False)
    later = []
    monkeypatch.setattr("typeless_local.app.AppHelper.callLater", lambda delay, fn: later.append((delay, fn)))
    app = TypelessLocalApp.__new__(TypelessLocalApp)
    overlay = _attach(app)
    app.hotkeys = _FailingHotkeys()
    app._start_model_prefetch = lambda: None
    app._prime_microphone = lambda: None

    app.start()

    assert overlay.calls == [("setup",), ("handle", True), ("show", "perm", {})]
    retry = [fn for delay, fn in later if fn == app._retry_hotkeys]
    assert retry, "a retry is scheduled while Accessibility is missing"

    # Still missing: it keeps looking.
    later.clear()
    app._retry_hotkeys()
    assert [fn for _, fn in later] == [app._retry_hotkeys]

    # Never tries the tap untrusted: that pops macOS's own prompt before the guide.
    tried = []
    app.hotkeys.start = lambda: tried.append(True)
    app._retry_hotkeys()
    assert tried == []

    # Granted: the tap goes in and the permission capsule goes away.
    monkeypatch.setattr("typeless_local.app.has_accessibility_trust", lambda: True)
    later.clear()
    app._retry_hotkeys()
    assert later == []
    assert overlay.calls[-1] == ("hide",)


def test_recording_timeout_finishes_and_submits_processing() -> None:
    app = _recording_app()
    app.state = "recording"
    app.focus_context = FocusContext(app_name="TextEdit", window_title="Untitled")

    app._finish_recording_after_timeout()

    assert app.state == "processing"
    assert app.recorder.stopped is True
    assert app.audio_ducker.calls == ["restore_all"]
    assert app.overlay.calls == [("show", "transcribing", {})]
    assert app.executor.submissions[0][0] == app._process_audio


def test_recording_ducks_system_audio_until_finish(monkeypatch) -> None:
    monkeypatch.setattr(
        "typeless_local.app.capture_focus_context",
        lambda **_: FocusContext(app_name="TextEdit", window_title="Untitled"),
    )
    app = _recording_app()

    app._start_recording("tap")
    app._finish_recording()

    assert app.audio_ducker.calls == ["duck", "restore_all"]
    assert app.recorder.started is True
    assert app.recorder.stopped is True
    assert app.overlay.shown() == ["starting", "rec", "transcribing"]


def test_qwen_starts_with_his_words_and_languages_then_reads_the_focused_app(monkeypatch) -> None:
    monkeypatch.setattr("typeless_local.app.capture_focus_context", lambda **_: FocusContext("Claude", "Jev 工具路由"))
    monkeypatch.setattr(app_module.languages, "known", lambda: True)
    monkeypatch.setattr(app_module.languages, "allowed", lambda heard: {"zh", "en"})
    stream = SimpleNamespace(feed=lambda chunk: None, cancel=lambda: None, context=None)
    stream.give_context = lambda texts: setattr(stream, "context", list(texts))
    opened = []
    monkeypatch.setattr(app_module.cloud_asr, "stream_if_set_up", lambda terms, rate, **kw: opened.append((terms, kw)) or stream)
    app = _recording_app()
    app.prefs = Preferences(asr_engine="qwen")
    app.vocab = ["Jev"]
    app.hotword_weights = {"Jev": 50}

    app._start_recording("tap")

    ((terms, kw),) = opened
    assert terms == ["Jev"] and kw["weights"] == {"Jev": 50} and kw["languages"] == ["en", "zh"] and kw["context_pending"]
    assert stream.context == ["Jev 工具路由"]


def test_a_stuck_microphone_fails_fast_and_offers_a_relaunch(monkeypatch) -> None:
    monkeypatch.setattr("typeless_local.app.capture_focus_context", lambda **_: FocusContext("TextEdit", "Untitled"))
    refreshed = []
    monkeypatch.setattr(app_module.devices, "refresh", lambda: refreshed.append("refresh"))
    monkeypatch.setattr(app_module.devices, "refresh_if_changed", lambda: refreshed.append("if_changed"))
    app = _recording_app()
    app.recorder.fail_start = True
    app.recorder.wedged = True

    app._start_recording("tap")

    assert refreshed == [], "PortAudio is not touched again while a stream is stuck"
    assert app.state == "idle"
    assert app.overlay.calls[-1] == ("show", "mic", {"why": "stuck"})


def test_recording_restores_audio_when_microphone_start_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "typeless_local.app.capture_focus_context",
        lambda **_: FocusContext(app_name="TextEdit", window_title="Untitled"),
    )
    app = _recording_app()
    app.recorder.fail_start = True

    app._start_recording("tap")

    # The mic opens before the speakers are muted, so a mic that never opens
    # leaves them alone; the restore on the error path is then a no-op.
    assert app.state == "idle"
    assert app.audio_ducker.calls == ["restore_all"]
    assert app.overlay.calls[-1] == ("show", "mic", {"why": "busy"})


def test_a_denied_microphone_is_reported_instead_of_recording_silence(monkeypatch) -> None:
    monkeypatch.setattr(app_module.permissions, "microphone_status", lambda: "denied")
    app = _recording_app()

    app._start_recording("tap")

    assert app.recorder.started is False
    assert app.state == "idle"
    assert app.overlay.calls == [("show", "mic", {"why": "denied"})]


def test_recording_restores_audio_when_microphone_stop_fails() -> None:
    app = _recording_app()
    app._countdown_timer = None
    app._finish_debounce_timer = None
    app.state = "recording"
    app.recorder.fail_stop = True
    app.focus_context = FocusContext(app_name="TextEdit", window_title="Untitled")

    app._finish_recording()

    assert app.state == "idle"
    assert app.audio_ducker.calls == ["restore_all"]
    assert app.executor.submissions == []
    assert app.overlay.calls[-1] == ("show", "error", {"msg": "麦克风出错"})


def _hotkey_app(monkeypatch):
    monkeypatch.setattr(
        "typeless_local.app.capture_focus_context",
        lambda **_: FocusContext(app_name="TextEdit", window_title="Untitled"),
    )
    app = _recording_app()
    app._countdown_timer = None
    app.hotkeys = _FakeHotkeys()
    clock = [0.0]
    monkeypatch.setattr("typeless_local.app.time.monotonic", lambda: clock[0])
    return app, app.hotkeys, clock


def test_primary_down_up_finishes_hold_to_talk(monkeypatch) -> None:
    app, hotkeys, clock = _hotkey_app(monkeypatch)

    clock[0] = hotkeys.last_primary_down_at = 10.0
    app._on_hotkey("primary_down")
    clock[0] = hotkeys.last_primary_up_at = 10.7
    app._on_hotkey("primary_up")

    assert app.state == "processing"
    assert app.recorder.started is True
    assert app.recorder.stopped is True
    assert app.executor.submissions
    assert app._hold_timer is None


def test_holding_the_key_switches_the_capsule_to_release_to_finish(monkeypatch) -> None:
    app, hotkeys, clock = _hotkey_app(monkeypatch)
    timers = []
    monkeypatch.setattr(
        "typeless_local.app.threading.Timer",
        lambda delay, callback: timers.append(_ManualTimer(delay, callback)) or timers[-1],
    )

    clock[0] = hotkeys.last_primary_down_at = 10.0
    app._on_hotkey("primary_down")
    (hold,) = [t for t in timers if t.delay == app_module.LONG_PRESS_SECONDS]
    clock[0] = 10.6
    hold.fire()  # still held: no key up since the key down

    assert app.overlay.calls[-1][1] == "rec" and app.overlay.calls[-1][2]["mode"] == "hold"
    assert app.overlay.calls[-1][2]["elapsed"] == 0.6


def test_hold_view_does_not_appear_after_a_quick_tap(monkeypatch) -> None:
    app, hotkeys, clock = _hotkey_app(monkeypatch)
    timers = []
    monkeypatch.setattr(
        "typeless_local.app.threading.Timer",
        lambda delay, callback: timers.append(_ManualTimer(delay, callback)) or timers[-1],
    )

    clock[0] = hotkeys.last_primary_down_at = 20.0
    app._on_hotkey("primary_down")
    clock[0] = hotkeys.last_primary_up_at = 20.1
    app._on_hotkey("primary_up")
    for timer in timers:
        timer.fire()

    assert app.state == "recording"
    assert [call[2]["mode"] for call in app.overlay.calls if call[:2] == ("show", "rec")] == ["click"]


def test_short_tap_release_keeps_recording_until_next_press(monkeypatch) -> None:
    app, hotkeys, clock = _hotkey_app(monkeypatch)

    clock[0] = hotkeys.last_primary_down_at = 30.0
    app._on_hotkey("primary_down")
    clock[0] = hotkeys.last_primary_up_at = 30.1
    app._on_hotkey("primary_up")

    assert app.state == "recording"
    assert app.recorder.stopped is False
    assert app.executor.submissions == []

    clock[0] = hotkeys.last_primary_down_at = 31.0
    app._on_hotkey("primary_down")

    assert app.state == "processing"
    assert app.recorder.stopped is True
    assert app.executor.submissions


def test_short_double_press_upgrades_to_hands_free(monkeypatch) -> None:
    app, hotkeys, clock = _hotkey_app(monkeypatch)

    clock[0] = hotkeys.last_primary_down_at = 20.0
    app._on_hotkey("primary_down")
    clock[0] = hotkeys.last_primary_up_at = 20.08
    app._on_hotkey("primary_up")
    clock[0] = hotkeys.last_primary_down_at = 20.24
    app._on_hotkey("primary_down")

    assert app.state == "recording"
    assert app.mode == "hands_free"
    assert app.overlay.calls[-1][1] == "rec" and app.overlay.calls[-1][2]["mode"] == "latch"
    assert app.executor.submissions == []


def test_finish_defers_until_microphone_delivers_first_chunk(monkeypatch) -> None:
    timers = []

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.daemon = False
            self.cancelled = False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr("typeless_local.app.threading.Timer", FakeTimer)

    app = _recording_app()
    app._countdown_timer = None
    app._finish_debounce_timer = None
    app._active_session_id = 1
    app.state = "recording"
    app.recorder.elapsed = 0.40
    app.recorder.chunk_count = 0

    app._finish_recording()

    assert app.state == "recording"
    assert app.recorder.stopped is False
    assert app.executor.submissions == []
    assert timers
    assert round(timers[0].delay, 2) == 0.35

    app.recorder.elapsed = 0.80
    timers[0].callback()

    assert app.state == "processing"
    assert app.recorder.stopped is True
    assert app.executor.submissions


def test_stale_empty_result_does_not_hide_new_recording() -> None:
    app = _make_app("")
    app.state = "processing"
    app._active_session_id = 1

    app._show_empty_then_idle(session_id=1)
    app.capsule.show("rec", mode="click")  # a new dictation starts before "empty" goes away
    for timer in _ManualTimer.made:
        timer.fire()

    assert app.overlay.calls == [("show", "empty", {"device": ""}), ("show", "rec", {"mode": "click"})]


def test_cancel_while_recording_says_so() -> None:
    app = _recording_app()
    app.state = "recording"

    app._cancel()

    assert app.state == "idle"
    assert app.recorder.stopped is True
    assert app.overlay.calls == [("show", "cancelled", {})]
    assert _ManualTimer.made[-1].delay == 0.8


class _FailingRefiner:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def refine(self, text, context, vocab=None, language=""):
        raise self.exc


def test_refine_failure_pastes_the_raw_transcript(monkeypatch) -> None:
    pasted = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("我们明天下午三点开会")
    app.asr = _FakeASR("我们明天下午三点开会", language="zh")
    app.refiner = _FailingRefiner(TimeoutError("refine timed out"))

    app._process_audio(
        np.ones(16000, dtype=np.float32),
        FocusContext("TextEdit", "Untitled", can_insert_text=True),
    )

    assert pasted == ["我们明天下午三点开会"]
    assert app.overlay.calls[-1] == ("show", "inserted-raw-net", {"why": "timeout"})


def test_own_key_out_of_credit_says_so_plainly(monkeypatch) -> None:
    import openai

    monkeypatch.setattr("typeless_local.app.paste_text", lambda text, **kwargs: None)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("hello there")
    app.asr = _FakeASR("hello there", language="en")
    response = SimpleNamespace(status_code=429, headers={}, request=None)
    exc = openai.RateLimitError(
        "no credits", response=response, body={"code": "credit_balance_exhausted", "type": "insufficient_quota"}
    )
    app.refiner = _FailingRefiner(exc)

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("TextEdit", "Untitled", can_insert_text=True))

    assert app.overlay.calls[-1] == ("show", "inserted-raw-net", {"why": "credit"})


def test_a_spent_trial_is_remembered_and_not_asked_again(monkeypatch) -> None:
    from typeless_local.refine import TrialUnavailable

    monkeypatch.setattr("typeless_local.app.paste_text", lambda text, **kwargs: None)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("hello there")
    app.asr = _FakeASR("hello there", language="en")
    app.prefs = Preferences()
    app.config.refine = SimpleNamespace(preset=app_module.trial.PRESET, model="m", api_key_env="YANA_TRIAL_TOKEN")
    saved = []
    app.set_preference = lambda key, value: (saved.append((key, value)), setattr(app, "prefs", dataclasses.replace(app.prefs, **{key: value})))
    app.refiner = _FailingRefiner(TrialUnavailable("trial_used_up"))

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("TextEdit", "Untitled", can_insert_text=True))
    assert saved == [("trial_used_up", True)]

    app.refiner = _FailingRefiner(AssertionError("the trial server must not be asked again"))
    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("TextEdit", "Untitled", can_insert_text=True))
    assert app.overlay.calls[-1][1] == "inserted-raw-trial"


def test_missing_key_pastes_raw_and_offers_to_set_one(monkeypatch) -> None:
    pasted = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("hello there")
    app.refiner = _FailingRefiner(MissingAPIKey("OPENAI_API_KEY is required"))

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("Notes", "", can_insert_text=True))

    assert pasted == ["hello there"]
    assert app.overlay.calls[-1] == ("show", "inserted-raw-key", {})
    assert _ManualTimer.made[-1].delay == 8.0


def test_used_up_trial_pastes_raw_and_asks_for_their_own_key(monkeypatch) -> None:
    from typeless_local.refine import TrialUnavailable

    pasted = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("hello there")
    app.refiner = _FailingRefiner(TrialUnavailable("trial_used_up"))

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("Notes", "", can_insert_text=True))

    assert pasted == ["hello there"]
    assert app.overlay.calls[-1] == ("show", "inserted-raw-trial", {"why": "trial_used_up"})
    assert app._trial_over is True


def test_refinement_switched_off_pastes_the_transcript_as_heard(monkeypatch) -> None:
    pasted = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("straight from whisper")
    app.prefs = Preferences(refine=False)

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("Notes", "", can_insert_text=True))

    assert app.refiner.calls == []
    assert pasted == ["straight from whisper"]
    assert app.overlay.shown() == ["inserted"]


def test_privacy_settings_limit_what_refinement_sees() -> None:
    app = _make_app("改一下")
    context = FocusContext(
        "Mail", "Re: 报价", selected_text="原来的句子", can_insert_text=True, before_text="王总您好，"
    )

    app.prefs = Preferences(send_before_text=True)
    assert app._refine_context(context) is context
    app.prefs = Preferences(rewrite_selection=False, send_window_title=False, send_before_text=False)
    assert app._refine_context(context) == FocusContext("", "", selected_text="", can_insert_text=True)


def test_short_confident_english_is_kept(monkeypatch) -> None:
    pasted = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("OK.")
    app.asr = _FakeASR("OK.", language="en", confidence=0.82)

    app._process_audio(
        np.ones(16000, dtype=np.float32),
        FocusContext("Slack", "#general", can_insert_text=True),
    )

    assert app.refiner.calls
    assert pasted == ["Refined text."]


def test_short_fragment_in_another_language_is_still_dropped() -> None:
    app = _make_app("はい")
    app.asr = _FakeASR("はい", language="ja", confidence=0.9)

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("", ""))

    assert app.refiner.calls == []


def _inserted_app(text: str = "Refined text.") -> TypelessLocalApp:
    app = _make_app("raw dictation")
    app.state = "idle"
    app.executor = _FakeExecutor()
    app._active_session_id = 3
    context = FocusContext("TextEdit", "Untitled", can_insert_text=True)
    app._insertion = Insertion(text, "raw dictation", TARGET_PID, context)
    app._keys_wanted = True
    app.capsule.show("inserted", n=2, replaced=False)
    return app


def test_undo_button_takes_the_paste_back_in_the_same_app(monkeypatch) -> None:
    undos = []
    monkeypatch.setattr(app_module, "undo_last_edit", lambda: undos.append("cmd-z"))
    app = _inserted_app()

    app._on_overlay_action("undo")

    assert undos == ["cmd-z"]
    assert app.overlay.calls[-1] == ("show", "undone", {})
    assert app._insertion is None and app._keys_wanted is False


def test_undo_button_refuses_once_another_app_is_in_front(monkeypatch) -> None:
    undos = []
    monkeypatch.setattr(app_module, "undo_last_edit", lambda: undos.append("cmd-z"))
    monkeypatch.setattr(app_module, "frontmost_pid", lambda: 999)
    app = _inserted_app()

    app._on_overlay_action("undo")

    assert undos == []
    assert app.overlay.calls[-1] == ("show", "notice", {"msg": "目标 App 已切换，没法撤销"})


def test_typing_after_the_paste_takes_the_undo_offer_away() -> None:
    app = _inserted_app()

    app._on_hotkey("typed")

    assert app.overlay.calls[-1] == ("hide",)
    assert app._insertion is None and app._keys_wanted is False


def test_the_users_own_cmd_z_is_confirmed() -> None:
    app = _inserted_app()

    app._on_hotkey("undo")

    assert app.overlay.calls[-1] == ("show", "undone", {})
    assert app._insertion is None


def test_cmd_z_in_another_app_is_not_taken_for_undoing_the_paste(monkeypatch) -> None:
    app = _inserted_app()
    monkeypatch.setattr(app_module, "frontmost_pid", lambda: TARGET_PID + 1)

    app._on_hotkey("undo")

    assert app.overlay.calls[-1] == ("hide",)
    assert app._insertion is None and app._keys_wanted is False


def test_choosing_english_in_settings_relocalizes_the_app(monkeypatch) -> None:
    from typeless_local import i18n

    app = _inserted_app()
    redrawn = []
    app.menubar = SimpleNamespace(set_state=lambda state: redrawn.append(("menubar", state)))
    app.windows = SimpleNamespace(relocalize=lambda: redrawn.append(("windows",)))
    app.overlay.relocalize = lambda: redrawn.append(("overlay",))
    monkeypatch.setattr(app_module, "install_main_menu", lambda: redrawn.append(("main menu",)))

    app.set_preference("ui_language", "en")

    assert i18n.current() == "en"
    assert {entry[0] for entry in redrawn} == {"main menu", "menubar", "overlay", "windows"}


def test_a_second_copy_of_the_app_cannot_take_the_lock(tmp_path) -> None:
    first = app_module.claim_single_instance(tmp_path)
    try:
        assert first is not None
        assert app_module.claim_single_instance(tmp_path) is None
    finally:
        first.close()
    again = app_module.claim_single_instance(tmp_path)
    assert again is not None
    again.close()


def _run_threads_inline(monkeypatch) -> None:
    monkeypatch.setattr(
        "typeless_local.app.threading.Thread",
        lambda target, args=(), **kwargs: SimpleNamespace(start=lambda: target(*args)),
    )
    monkeypatch.setattr("typeless_local.app.time.sleep", lambda seconds: None)


def test_replace_takes_the_paste_back_and_pastes_the_edit(monkeypatch) -> None:
    events = []
    _run_threads_inline(monkeypatch)
    monkeypatch.setattr(app_module, "undo_last_edit", lambda: events.append("cmd-z"))
    monkeypatch.setattr("typeless_local.app.paste_text", lambda text: events.append(("paste", text)))
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _inserted_app()
    app._on_overlay_action("edit")
    assert app.overlay.calls[-1] == ("show", "edit-modify", {"text": "Refined text."})

    app._on_overlay_action("replace", {"text": "  改好的文字 "})

    assert events == ["cmd-z", ("paste", "改好的文字")]
    assert app.overlay.calls[-2:] == [("end_edit",), ("show", "replaced", {"n": 5})]
    assert app._insertion.text == "改好的文字"


def test_replace_after_typing_elsewhere_copies_instead_of_undoing(monkeypatch) -> None:
    events, copied = [], []
    _run_threads_inline(monkeypatch)
    monkeypatch.setattr(app_module, "undo_last_edit", lambda: events.append("cmd-z"))
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", copied.append)
    app = _inserted_app()
    app._on_overlay_action("edit")
    app._on_overlay_action("field", {"focus": False})  # clicked back into the other app
    app._on_hotkey("typed")

    app._on_overlay_action("replace", {"text": "改好的文字"})

    assert events == []
    assert copied == ["改好的文字"]
    assert app.overlay.calls[-1] == ("show", "notice", {"msg": "原文已经改动过，修改后的文字已复制"})


def test_rerefine_replaces_the_raw_paste_with_the_refined_text(monkeypatch) -> None:
    events = []
    _run_threads_inline(monkeypatch)
    monkeypatch.setattr(app_module, "undo_last_edit", lambda: events.append("cmd-z"))
    monkeypatch.setattr("typeless_local.app.paste_text", lambda text: events.append(("paste", text)))
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _inserted_app("raw dictation")
    app.capsule.show("inserted-raw-net", why="timeout")

    app._on_overlay_action("rerefine")
    fn, args = app.executor.submissions[-1]
    fn(*args)

    assert app.refiner.calls[-1][0] == "raw dictation"
    assert events == ["cmd-z", ("paste", "Refined text.")]
    assert app.overlay.shown()[-2:] == ["refining", "replaced"]


def test_rerefine_that_fails_again_leaves_the_raw_text(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(app_module, "undo_last_edit", lambda: events.append("cmd-z"))
    app = _inserted_app("raw dictation")
    app.refiner = _FailingRefiner(TimeoutError("timed out again"))

    app._on_overlay_action("rerefine")
    fn, args = app.executor.submissions[-1]
    fn(*args)

    assert events == []
    assert app.overlay.calls[-1] == ("show", "inserted-raw-net", {"why": "timeout"})


def test_card_draft_keeps_the_clipboard_current_only_where_nothing_was_pasted(monkeypatch) -> None:
    copied = []
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", copied.append)
    app = _make_app("x")
    app.capsule.show("edit-notarget", text="第一版")
    app._on_overlay_action("draft", {"text": "第二版"})
    app.capsule.show("edit-modify", text="已粘贴的")
    app._on_overlay_action("draft", {"text": "不该进剪贴板"})

    assert copied == ["第二版"]


def test_done_in_the_card_copies_the_edit(monkeypatch) -> None:
    copied = []
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", copied.append)
    app = _make_app("x")
    app.config = SimpleNamespace(user_paths=None)
    app._copy_fallback_text = "提醒我周五"

    app._on_overlay_action("done", {"text": "提醒我周五之前续证书"})

    assert copied == ["提醒我周五之前续证书"]
    assert app.overlay.calls[-2:] == [("end_edit",), ("show", "copied", {})]


def test_hovering_holds_the_capsule_until_the_pointer_leaves() -> None:
    app = _inserted_app()
    (timer,) = _ManualTimer.made
    app._on_overlay_hover(True)
    assert timer.cancelled

    app._on_overlay_hover(False)
    resumed = _ManualTimer.made[-1]
    assert resumed is not timer and resumed.delay >= 1.2
    resumed.fire()
    assert app.overlay.calls[-1] == ("hide",)


def test_download_in_progress_blocks_recording(monkeypatch) -> None:
    app = _recording_app()
    app._download = (0.5, 0.0)
    monkeypatch.setattr("typeless_local.app.time.monotonic", lambda: 20.0)

    app._start_recording("tap")

    assert app.recorder.started is False
    assert app.overlay.calls == [("show", "download", {"p": 0.5, "eta": "约 20 秒"})]


def test_download_progress_never_replaces_a_dictation(monkeypatch) -> None:
    monkeypatch.setattr("typeless_local.app.model_is_cached", lambda repo: False)
    monkeypatch.setattr(
        "typeless_local.app.threading.Thread",
        lambda target, **kwargs: SimpleNamespace(start=target),
    )
    app = _recording_app()
    app.config = SimpleNamespace(jarvis_config={"asr": {"provider": "mlx_whisper"}})
    app.asr = SimpleNamespace(warmup=lambda: None)

    def download(repo, report):
        app.state = "recording"
        app.capsule.show("rec", mode="click")
        report(0.4)  # mid-dictation: must not show
        app.state = "idle"
        app.capsule.hide()
        report(0.8)

    monkeypatch.setattr("typeless_local.app.download_model", download)

    app._start_model_prefetch()

    assert app.overlay.shown() == ["download", "rec", "download", "ready"]  # "ready" replaces the progress
    assert app._download is None


def test_mic_opens_before_focus_probe_and_mute_runs_off_the_hotkey_path(monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor

    order = []
    monkeypatch.setattr(
        "typeless_local.app.capture_focus_context",
        lambda **_: order.append("focus") or FocusContext(app_name="TextEdit", window_title="Untitled"),
    )
    app = _recording_app()
    original_start = app.recorder.start
    app.recorder.start = lambda: (order.append("mic"), original_start())
    release = threading.Event()

    class SlowDucker(_FakeDucker):
        def duck(self) -> bool:
            release.wait(2)
            return super().duck()

    app.audio_ducker = SlowDucker()
    app._audio_io = ThreadPoolExecutor(max_workers=1)

    app._start_recording("tap")
    # The slow mute has not finished, yet the recording is already running.
    assert order == ["mic", "focus"]
    assert app.state == "recording"
    app._finish_recording()
    release.set()
    app._audio_io.shutdown(wait=True)

    assert app.audio_ducker.calls == ["duck", "restore_all"]


def test_capsule_follows_the_caret_when_asked(monkeypatch) -> None:
    monkeypatch.setattr(
        "typeless_local.app.capture_focus_context",
        lambda **_: FocusContext(app_name="TextEdit", window_title="Untitled"),
    )
    monkeypatch.setattr(app_module, "caret_rect", lambda: (400.0, 600.0, 2.0, 18.0))
    app = _recording_app()
    app.prefs = Preferences(capsule_position="caret")

    app._start_recording("tap")

    # No "starting" at the bottom first: the capsule appears at the caret.
    kinds = [call[0] if call[0] != "show" else call[1] for call in app.overlay.calls]
    assert kinds == ["anchor", "rec"]
    assert app.overlay.calls[0] == ("anchor", "caret", (400.0, 600.0, 2.0, 18.0))


def test_prefetch_warms_the_recognizer_when_weights_are_cached(monkeypatch) -> None:
    monkeypatch.setattr("typeless_local.app.model_is_cached", lambda repo: True)
    app = TypelessLocalApp.__new__(TypelessLocalApp)
    app.config = SimpleNamespace(jarvis_config={"asr": {"provider": "mlx_whisper"}})
    app.asr = SimpleNamespace(warmup=lambda: None)
    app.executor = _FakeExecutor()

    app._start_model_prefetch()

    assert app.executor.submissions == [(app.asr.warmup, ())]


def test_refine_prewarm_runs_at_most_once_every_few_seconds(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("typeless_local.app.time.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "typeless_local.app.threading.Thread",
        lambda target, **kwargs: SimpleNamespace(start=target),
    )
    app = _make_app("hi")
    prewarms = []
    app.refiner.prewarm = lambda: prewarms.append(clock[0])

    app._prewarm_refiner()  # recording starts
    clock[0] += 1.0
    app._prewarm_refiner()  # a stretch is cut a second later: still warm
    clock[0] += 3.0
    app._prewarm_refiner()  # four seconds in: warm it again before it idles out

    assert prewarms == [100.0, 104.0]


def test_count_units_counts_chinese_characters_and_latin_words() -> None:
    assert count_units("明天下午四点跟设计组过一下。") == 13
    assert count_units("Refined text.") == 2
    assert count_units("过一下 Typlus 的新版浮窗，then ship it") == 12


# ------------------------------------------------------------- menu bar


class _FakeMenubar:
    def __init__(self) -> None:
        self.issues = None
        self.states = []

    def set_issues(self, issues) -> None:
        self.issues = tuple(issues)

    def set_state(self, state) -> None:
        self.states.append(state)


def _menu_app(monkeypatch, env=None) -> TypelessLocalApp:
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(app_module, "has_accessibility_trust", lambda: True)
    app = _recording_app()
    app.menubar = _FakeMenubar()
    jarvis = {
        "llm": {
            "default_preset": "mini",
            "presets": {
                "mini": {"model": "gpt-mini", "api_key_env": "TEST_KEY_A"},
                "deep": {"model": "deepseek-chat", "api_key_env": "TEST_KEY_B"},
            },
        }
    }
    app.config = SimpleNamespace(
        jarvis_config=jarvis,
        refine=app_module.refine_config_for(jarvis, "mini"),
        input_device="",
        user_paths=None,
        sample_rate=16000,
        min_recording_seconds=0.25,
        max_recording_seconds=0,
    )
    return app


def test_missing_key_accessibility_and_mic_badge_the_icon(monkeypatch) -> None:
    monkeypatch.delenv("TEST_KEY_A", raising=False)
    app = _menu_app(monkeypatch)
    assert app._refresh_issues() == ("key",)
    assert app.menubar.issues == ("key",)

    monkeypatch.setenv("TEST_KEY_A", "sk-test")
    monkeypatch.setattr(app_module, "has_accessibility_trust", lambda: False)
    monkeypatch.setattr(app_module.permissions, "microphone_status", lambda: "denied")
    assert app._refresh_issues() == ("perm", "mic")


def test_refine_off_needs_no_key(monkeypatch) -> None:
    monkeypatch.delenv("TEST_KEY_A", raising=False)
    app = _menu_app(monkeypatch)
    app.prefs = Preferences(refine=False)
    assert app._current_issues() == ()


def test_menu_snapshot_reads_the_model_devices_and_the_last_dictation(monkeypatch) -> None:
    monkeypatch.delenv("TEST_KEY_B", raising=False)
    app = _menu_app(monkeypatch, env={"TEST_KEY_A": "sk-test"})
    monkeypatch.setattr(app_module.devices, "refresh_if_changed", lambda: False)
    monkeypatch.setattr(app_module.devices, "list_input_devices", lambda: ["MacBook Pro 麦克风"])
    app._remember("明天开会", "备忘录")

    snap = app._menu_snapshot()

    assert snap.state == "idle" and snap.issues == ()
    assert snap.active_preset == "mini"
    assert snap.inputs == ("MacBook Pro 麦克风",)
    assert (snap.recent.text, snap.recent.app) == ("明天开会", "备忘录")
    assert app._copy_fallback_text == "明天开会"


def test_menu_actions_reach_the_app(monkeypatch) -> None:
    app = _menu_app(monkeypatch, env={"TEST_KEY_A": "sk-test"})
    calls = []
    app._start_recording = lambda mode: calls.append(("start", mode))
    app.select_model = lambda name: calls.append(("model", name))
    app.select_input_device = lambda name: calls.append(("input", name))
    app.open_settings = lambda pane=None: calls.append(("settings", pane))
    app.open_history = lambda: calls.append(("history",))
    monkeypatch.setattr(app_module.permissions, "open_url", lambda url: calls.append(("url", url)))
    monkeypatch.setattr(app_module, "request_accessibility_trust", lambda: calls.append(("ask-ax",)))

    for key in ("toggle", "latch", "preset:deep", "input:", "settings:", "settings:vocab", "history", "fix:key", "fix:perm"):
        app._on_menu_action(key)

    assert calls == [
        ("start", "tap"),
        ("start", "hands_free"),
        ("model", "deep"),
        ("input", ""),
        ("settings", None),
        ("settings", "vocab"),
        ("history",),
        ("settings", "model"),
        ("ask-ax",),
        ("url", app_module.permissions.ACCESSIBILITY_SETTINGS),
    ]


def test_menu_toggle_finishes_a_recording(monkeypatch) -> None:
    app = _menu_app(monkeypatch, env={"TEST_KEY_A": "sk-test"})
    finished = []
    app.state = "recording"
    app._finish_recording = lambda: finished.append(True)
    app._on_menu_action("toggle")
    app._on_menu_action("latch")  # only starts from idle
    assert finished == [True]


def test_menu_copy_puts_the_last_dictation_back(monkeypatch) -> None:
    copied = []
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", copied.append)
    app = _menu_app(monkeypatch, env={"TEST_KEY_A": "sk-test"})
    app._remember("上一段话", "Notes")
    app._on_menu_action("copy")
    assert copied == ["上一段话"]
    assert app.overlay.calls[-1] == ("show", "copied", {})


def test_a_failed_dictation_leaves_the_icon_idle(monkeypatch) -> None:
    app = _make_app("hello there friend")
    app.menubar = _FakeMenubar()
    app.headless = False
    app.state = "processing"
    app._active_session_id = 1
    app.refiner.refine = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setattr("typeless_local.app.paste_text", lambda text: (_ for _ in ()).throw(RuntimeError("no paste")))
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("TextEdit", "x", can_insert_text=True), 1)

    assert app.menubar.states[-1] == "idle"
    assert app.overlay.calls[-1][:2] == ("show", "error")


def test_recording_start_queues_a_whisper_warm_up_ahead_of_the_audio(monkeypatch) -> None:
    monkeypatch.setattr(
        "typeless_local.app.capture_focus_context",
        lambda **_: FocusContext(app_name="TextEdit", window_title="Untitled"),
    )
    app = TypelessLocalApp.__new__(TypelessLocalApp)
    app._lock = threading.RLock()
    app._recording_timer = None
    app.config = SimpleNamespace(sample_rate=16000, min_recording_seconds=0.25, max_recording_seconds=0)
    app.state = "idle"
    app.mode = "tap"
    _attach(app)
    app.recorder = _FakeRecorder()
    app.audio_ducker = _FakeDucker()
    app.executor = _FakeExecutor()
    app.asr = SimpleNamespace(warmup=lambda: None)

    app._start_recording("tap")

    assert [fn for fn, _args in app.executor.submissions] == [app.asr.warmup]


def test_microphone_is_primed_at_launch_under_the_hotkey_lock(monkeypatch) -> None:
    app = TypelessLocalApp.__new__(TypelessLocalApp)
    app._lock = threading.RLock()
    app.config = SimpleNamespace(sample_rate=16000)
    app.recorder = SimpleNamespace(device=2)
    primed = []
    app._select_capture_device = lambda: "reSpeaker"
    monkeypatch.setattr(
        "typeless_local.app.devices.prime_input",
        lambda index, rate: primed.append((index, rate, app._lock._is_owned())),
    )
    monkeypatch.setattr(
        "typeless_local.app.threading.Thread",
        lambda target, **kwargs: SimpleNamespace(start=target),
    )

    app._prime_microphone()

    assert primed == [(2, 16000, True)]


# ------------------------------------------------------------- windows


class _FakeWindows:
    def __init__(self) -> None:
        self.calls = []

    def show_onboarding(self) -> None:
        self.calls.append(("onboarding",))

    def refresh(self, history=False) -> None:
        self.calls.append(("refresh", history))

    def download(self, fraction, eta="", done=False, error=False, why="") -> None:
        self.calls.append(("download", fraction, done, error))


def _windows_app(monkeypatch, env=None):
    app = _menu_app(monkeypatch, env)
    app.windows = _FakeWindows()
    app.prefs = app_module.Preferences()
    saved = []
    app.set_preference = lambda key, value: saved.append((key, value))
    app._saved = saved
    monkeypatch.setattr(app_module.permissions, "microphone_status", lambda: "authorized")
    monkeypatch.setattr(app_module, "model_is_cached", lambda repo: True)
    return app


def test_first_launch_opens_the_guide(monkeypatch) -> None:
    app = _windows_app(monkeypatch)  # no API key yet
    app._first_run()
    assert app.windows.calls == [("onboarding",)]
    assert app._saved == []


def test_first_launch_skips_the_guide_when_everything_is_set_up(monkeypatch) -> None:
    app = _windows_app(monkeypatch, {"TEST_KEY_A": "sk-test-aaaaaaaaaaaa1234"})
    app._first_run()
    assert app.windows.calls == []
    assert app._saved == [("onboarding_done", True), ("f5_hotkey", True)]


def test_the_guide_is_shown_only_once(monkeypatch) -> None:
    app = _windows_app(monkeypatch)
    app.prefs = app_module.Preferences(onboarding_done=True)
    app._first_run()
    assert app.windows.calls == []


def test_language_and_ducking_apply_live_and_persist(monkeypatch, tmp_path) -> None:
    app = _windows_app(monkeypatch)
    written = []
    monkeypatch.setattr(app_module, "save_user_setting", lambda paths, *args: written.append(args))
    app.config.user_paths = SimpleNamespace(env_path=tmp_path / "env")
    languages = []
    app.asr = SimpleNamespace(set_language=languages.append)

    app.set_language("en")
    app.set_language("fr")
    app.set_language("klingon")
    app.set_ducking(False)

    assert languages == ["en", "fr", ""]
    assert app.config.jarvis_config["asr"]["language"] == ""
    assert app.audio_ducker.enabled is False
    assert app.config.jarvis_config["audio_ducking"]["enabled"] is False
    assert written == [("asr", "language", "en"), ("asr", "language", "fr"), ("asr", "language", ""), ("audio_ducking", "enabled", False)]


def test_a_new_key_for_the_active_model_replaces_the_client(monkeypatch, tmp_path) -> None:
    app = _windows_app(monkeypatch)
    stored = []
    monkeypatch.setattr(app_module, "_store_api_key", lambda env, key, path: stored.append((env, key, path)))
    app.config.user_paths = SimpleNamespace(env_path=tmp_path / "env")
    old = app.refiner = object()

    app.store_api_key("TEST_KEY_A", "sk-test-aaaaaaaaaaaa1234")

    assert stored == [("TEST_KEY_A", "sk-test-aaaaaaaaaaaa1234", tmp_path / "env")]
    assert app.refiner is not old and app.refiner.config.api_key_env == "TEST_KEY_A"


def test_download_progress_reaches_the_guide(monkeypatch) -> None:
    app = _windows_app(monkeypatch)
    app._download_progress(0.4, "约 30 秒")
    app._download_progress(1.0, done=True)
    assert app.windows.calls == [("download", 0.4, False, False), ("download", 1.0, True, False)]


def test_switching_model_or_mic_redraws_settings(monkeypatch) -> None:
    app = _windows_app(monkeypatch)
    monkeypatch.setattr(app_module.devices, "resolve_input_index", lambda name: None)
    app.config = app_module.AppConfig(
        root=Path("."), jarvis_root=Path("."), jarvis_config=app.config.jarvis_config, refine=app.config.refine
    )
    app.select_model("deep")
    app.select_input_device("USB Mic")
    assert app.windows.calls == [("refresh", False), ("refresh", False)]


def test_extend_adds_fifteen_minutes_to_a_recording_near_its_limit(monkeypatch) -> None:
    monkeypatch.setattr(app_module.threading, "Timer", _ManualTimer)
    clock = [1000.0]
    monkeypatch.setattr(app_module.time, "monotonic", lambda: clock[0])
    app = _recording_app()
    app.config = SimpleNamespace(sample_rate=16000, min_recording_seconds=0.25, max_recording_seconds=900.0)
    app.prefs = app_module.Preferences()
    app.state = "recording"
    app._recording_started_at = 1000.0
    app._recording_limit_s = 900.0
    app._start_recording_timeout()
    (first,) = _ManualTimer.made
    assert first.delay == 900.0

    clock[0] += 870.0  # 30 seconds left: the countdown is up
    app._on_overlay_action("extend")

    assert first.cancelled
    second = _ManualTimer.made[-1]
    assert second.delay == 930.0 and not second.cancelled
    kind, state, data = app.overlay.calls[-1]
    assert (kind, state, data["max"], data["ext"]) == ("show", "rec", 1800.0, 15)
    assert data["elapsed"] == 870.0


def test_extend_does_nothing_once_the_recording_is_over(monkeypatch) -> None:
    monkeypatch.setattr(app_module.threading, "Timer", _ManualTimer)
    app = _recording_app()
    app.state = "processing"
    app._recording_limit_s = 900.0
    app._on_overlay_action("extend")
    assert _ManualTimer.made == [] and app.overlay.calls == []


def test_a_pasted_dictation_is_watched_until_it_is_sent(monkeypatch, tmp_path) -> None:
    from typeless_local.config import UserPaths
    from typeless_local.trace import DictationTrace
    import dataclasses

    from typeless_local import history

    monkeypatch.setattr("typeless_local.app.paste_text", lambda _text: None)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("raw dictation")
    db = tmp_path / "trace.db"
    app.config.user_paths = UserPaths(
        config_dir=tmp_path, vocab_path=tmp_path / "v.yaml", corrections_path=tmp_path / "c.yaml", trace_db_path=db,
        log_path=tmp_path / "a.log", stopwords_dir=tmp_path, user_config_path=tmp_path / "config.yaml", env_path=tmp_path / "env",
    )
    app.trace = DictationTrace(db)
    watched = []
    app.sent_watcher = SimpleNamespace(watch=lambda *args: watched.append(args), cancel=lambda: None)

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext(app_name="微信", window_title="", can_insert_text=True))

    ((row, pasted, pid),) = watched
    assert pasted == "Refined text." and pid == TARGET_PID
    app._store_sent_text(row, pasted, "Refined text, edited.")
    assert history.recent_sessions(db)[0]["sent"] == "Refined text, edited."

    # Off in Settings: nothing is watched.
    app.prefs = dataclasses.replace(app.prefs, save_sent_text=False)
    watched.clear()
    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext(app_name="微信", window_title="", can_insert_text=True))
    assert watched == []


def test_text_before_the_caret_is_only_read_when_the_setting_is_on(monkeypatch) -> None:
    asked = []
    monkeypatch.setattr(
        "typeless_local.app.capture_focus_context",
        lambda read_before_text=False: asked.append(read_before_text) or FocusContext("TextEdit", "Untitled"),
    )
    app = _recording_app()
    assert app.prefs.send_before_text is False  # off unless the user turns it on

    app._start_recording("tap")
    app._finish_recording()
    app.prefs = Preferences(send_before_text=True)
    app.state = "idle"
    app._start_recording("tap")
    app._finish_recording()

    assert asked == [False, True]


def test_music_playing_elsewhere_is_paused_for_the_recording_and_resumed(monkeypatch, _media_keys) -> None:
    monkeypatch.setattr(
        "typeless_local.app.capture_focus_context",
        lambda **_: FocusContext(app_name="TextEdit", window_title="Untitled"),
    )
    music, playing = 101, {101}
    monkeypatch.setattr(app_module.devices, "playing_apps", lambda: set(playing))
    monkeypatch.setattr(app_module, "press_play_pause", lambda: (_media_keys.append("play_pause"), playing.symmetric_difference_update({music})))
    app = _recording_app()

    app._start_recording("tap")
    assert _media_keys == ["play_pause"] and playing == set()  # it stopped sending audio: paused
    app._finish_recording()
    assert _media_keys == ["play_pause", "play_pause"] and playing == {music}


def test_a_player_that_keeps_output_open_is_put_back_as_it_was(monkeypatch, _media_keys) -> None:
    """NetEase holds output open while paused: no change is visible, so the press is undone."""

    monkeypatch.setattr("typeless_local.app.capture_focus_context", lambda **_: FocusContext("TextEdit", "Untitled"))
    monkeypatch.setattr(app_module, "MEDIA_CHECK_S", 0.02)
    monkeypatch.setattr(app_module.devices, "playing_apps", lambda: {101})
    monkeypatch.setattr(app_module.devices, "is_call_app", lambda pid: False)
    app = _recording_app()

    app._start_recording("tap")
    assert _media_keys == ["play_pause", "play_pause"]
    app._finish_recording()
    assert _media_keys == ["play_pause", "play_pause"]


def _unclear_app(monkeypatch, playing: set[int]):
    monkeypatch.setattr("typeless_local.app.capture_focus_context", lambda **_: FocusContext("TextEdit", "Untitled"))
    monkeypatch.setattr(app_module, "MEDIA_CHECK_S", 0.02)
    monkeypatch.setattr(app_module.devices, "playing_apps", lambda: set(playing))
    monkeypatch.setattr(app_module.devices, "bundle_id", lambda pid: {101: "com.netease.163music", 102: "com.other.player"}.get(pid, ""))
    app = _recording_app()
    app.prefs = Preferences()
    app.set_preference = lambda key, value: setattr(app, "prefs", dataclasses.replace(app.prefs, **{key: value}))
    return app


def test_an_unclear_press_remembers_the_player_and_it_is_not_pressed_again(monkeypatch, _media_keys) -> None:
    app = _unclear_app(monkeypatch, {101})

    app._start_recording("tap")
    assert _media_keys == ["play_pause", "play_pause"]  # pressed and undone this once
    assert app.prefs.unclear_media_apps == ["com.netease.163music"]
    app._finish_recording()

    _media_keys.clear()
    app._start_recording("tap")
    app._finish_recording()
    assert _media_keys == []


def test_a_second_player_still_gets_the_press_as_before(monkeypatch, _media_keys) -> None:
    app = _unclear_app(monkeypatch, {101, 102})
    app.prefs = Preferences(unclear_media_apps=["com.netease.163music"])

    app._start_recording("tap")
    assert _media_keys == ["play_pause", "play_pause"]
    assert app.prefs.unclear_media_apps == ["com.netease.163music", "com.other.player"]


def test_only_a_call_holding_output_never_touches_play_pause(monkeypatch, _media_keys) -> None:
    monkeypatch.setattr("typeless_local.app.capture_focus_context", lambda **_: FocusContext("TextEdit", "Untitled"))
    monkeypatch.setattr(app_module.devices, "playing_apps", lambda: {200})
    monkeypatch.setattr(app_module.devices, "is_call_app", lambda pid: pid == 200)
    app = _recording_app()

    app._start_recording("tap")
    app._finish_recording()
    assert _media_keys == []


def test_paused_music_is_never_started_by_the_recording(monkeypatch, _media_keys) -> None:
    """In class: Zoom holds output open, the music is paused, and play/pause would start it."""

    monkeypatch.setattr(
        "typeless_local.app.capture_focus_context",
        lambda **_: FocusContext(app_name="TextEdit", window_title="Untitled"),
    )
    zoom, music = 200, 101
    playing = {zoom}
    monkeypatch.setattr(app_module.devices, "playing_apps", lambda: set(playing))
    monkeypatch.setattr(app_module, "press_play_pause", lambda: (_media_keys.append("play_pause"), playing.symmetric_difference_update({music})))
    app = _recording_app()

    app._start_recording("tap")
    assert _media_keys == ["play_pause", "play_pause"]  # started the music, so stopped it again at once
    assert playing == {zoom}
    app._finish_recording()
    assert _media_keys == ["play_pause", "play_pause"]  # and nothing "resumed" afterwards
    assert playing == {zoom}


def test_music_is_left_alone_when_muting_other_sound_is_off(monkeypatch, _media_keys) -> None:
    monkeypatch.setattr(
        "typeless_local.app.capture_focus_context",
        lambda **_: FocusContext(app_name="TextEdit", window_title="Untitled"),
    )
    monkeypatch.setattr(app_module.devices, "playing_apps", lambda: {101})
    app = _recording_app()
    app.audio_ducker.enabled = False

    app._start_recording("tap")
    app._finish_recording()
    assert _media_keys == []


def _main_env(monkeypatch, tmp_path, *, chip: str, china: bool, onboarding_done: bool = False, own_key: bool = False, trial_region: bool = True):
    """Run main() up to the point it builds the app, with nothing real underneath."""

    from typeless_local import config as cfg_mod

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "jarvis").mkdir()
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path / "jarvis"))
    if onboarding_done:
        cfg_mod.save_user_setting(cfg_mod.resolve_user_paths(), "ui", "onboarding_done", True)
    # The alerts are checked in Chinese whatever language this Mac runs in.
    cfg_mod.save_user_setting(cfg_mod.resolve_user_paths(), "ui", "ui_language", "zh")
    seen = SimpleNamespace(alerts=[], built=[])
    # The real NSApplication.run() never returns.
    fake_app = SimpleNamespace(setActivationPolicy_=lambda policy: None, run=lambda: None)
    monkeypatch.setattr(app_module, "NSApplication", SimpleNamespace(sharedApplication=lambda: fake_app))
    monkeypatch.setattr(app_module.trial, "ensure_token", lambda: None)
    if own_key:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-aaaaaaaaaaaa")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(app_module, "configure_logging", lambda: None)
    monkeypatch.setattr(app_module, "claim_single_instance", lambda config_dir: object())
    monkeypatch.setattr(app_module.reach, "machine", lambda: chip)
    monkeypatch.setattr(app_module.reach, "in_mainland_china", lambda: china)
    monkeypatch.setattr(app_module.reach, "in_trial_region", lambda: trial_region and not china)
    monkeypatch.setattr(app_module, "_alert", lambda app, title, body: seen.alerts.append(title))
    monkeypatch.setattr(app_module.keychain, "fill_environ", lambda names: None)
    monkeypatch.setattr(
        app_module,
        "TypelessLocalApp",
        lambda config: seen.built.append(config) or SimpleNamespace(start=lambda: None),
    )
    app_module.main()
    return seen


def test_main_explains_and_stops_on_an_intel_mac(monkeypatch, tmp_path) -> None:
    seen = _main_env(monkeypatch, tmp_path, chip="intel", china=False)
    assert seen.built == []
    assert seen.alerts == ["言字需要 Apple 芯片的 Mac"]


def test_main_asks_to_turn_rosetta_off(monkeypatch, tmp_path) -> None:
    seen = _main_env(monkeypatch, tmp_path, chip="rosetta", china=False)
    assert seen.built == [] and seen.alerts == ["言字正在用 Rosetta 打开"]


def test_a_new_install_in_mainland_china_starts_on_deepseek(monkeypatch, tmp_path) -> None:
    (config,) = _main_env(monkeypatch, tmp_path, chip="apple", china=True).built
    assert config.refine.preset == app_module.reach.CHINA_PRESET


def test_an_install_that_finished_the_guide_keeps_its_model(monkeypatch, tmp_path) -> None:
    (config,) = _main_env(monkeypatch, tmp_path, chip="apple", china=True, onboarding_done=True).built
    assert config.refine.preset == "gpt-5.6-terra"


def test_a_new_install_elsewhere_with_no_key_starts_on_the_free_trial(monkeypatch, tmp_path) -> None:
    (config,) = _main_env(monkeypatch, tmp_path, chip="apple", china=False).built
    assert config.refine.preset == app_module.trial.PRESET


def test_a_new_install_outside_the_us_and_canada_is_asked_for_a_key(monkeypatch, tmp_path) -> None:
    (config,) = _main_env(monkeypatch, tmp_path, chip="apple", china=False, trial_region=False).built
    assert config.refine.preset != app_module.trial.PRESET


def test_a_new_install_with_its_own_key_skips_the_trial(monkeypatch, tmp_path) -> None:
    (config,) = _main_env(monkeypatch, tmp_path, chip="apple", china=False, own_key=True).built
    assert config.refine.preset == "gpt-5.6-terra"


def test_diagnostics_are_zipped_into_the_data_folder_without_keys(monkeypatch, tmp_path) -> None:
    import zipfile

    from typeless_local.config import UserPaths

    _run_threads_inline(monkeypatch)
    opened = []
    monkeypatch.setattr(app_module.subprocess, "Popen", lambda args: opened.append(args))
    monkeypatch.setattr(app_module.devices, "list_input_devices", lambda: ["USB Mic"])
    monkeypatch.setattr(app_module.devices, "current_input_device", lambda: "USB Mic")
    monkeypatch.setattr(app_module, "model_is_cached", lambda repo: True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdefghij")
    app = _recording_app()
    app.prefs = Preferences()
    paths = UserPaths(
        config_dir=tmp_path, vocab_path=tmp_path / "v.yaml", corrections_path=tmp_path / "c.yaml",
        trace_db_path=tmp_path / "trace.db", log_path=tmp_path / "app.log", stopwords_dir=tmp_path,
        user_config_path=tmp_path / "config.yaml", env_path=tmp_path / "env",
    )
    paths.log_path.write_text("using sk-test-0123456789abcdefghij\n")
    from typeless_local.config import refine_config_for

    jarvis = {"asr": {"provider": "mlx_whisper"}, "llm": {"presets": {"mini": {"model": "gpt-5.4-mini"}}}}
    app.config = SimpleNamespace(
        jarvis_config=jarvis, refine=refine_config_for(jarvis, "mini"), user_paths=paths, input_device="", input_channel=0
    )
    app._current_issues = lambda: ()

    app.export_diagnostics()

    ((_, flag, path),) = opened
    assert flag == "-R" and path.startswith(str(tmp_path / "diagnostics"))
    with zipfile.ZipFile(path) as archive:
        everything = b"".join(archive.read(name) for name in archive.namelist()).decode()
    assert "sk-test-0123456789abcdefghij" not in everything
    assert '"OPENAI_API_KEY": true' in everything and "USB Mic" in everything


def test_saving_an_openai_key_on_the_trial_switches_off_the_trial(monkeypatch, tmp_path) -> None:
    from typeless_local import config as cfg_mod, trial

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    config = cfg_mod.adopt_default_preset(cfg_mod.load_config(), trial.PRESET)
    assert config.refine.preset == trial.PRESET
    monkeypatch.setattr(app_module, "_store_api_key", lambda *args: None)
    app = TypelessLocalApp.__new__(TypelessLocalApp)
    app.config = config
    app._call_ui = lambda fn, *args: None
    app._windows_changed = lambda *args, **kwargs: None
    app._trial_over = True

    app.store_api_key("OPENAI_API_KEY", "sk-test-aaaaaaaaaaaa")

    assert app.config.refine.preset == trial.OWN_KEY_PRESET
    assert app.refiner.config.api_key_env == "OPENAI_API_KEY"
    assert app._trial_over is False
    assert "default_preset: gpt-5.6-terra" in (tmp_path / ".typlus" / "config.yaml").read_text()


def test_a_right_cmd_hold_records_until_release_and_ducks_only_once_it_lasts(monkeypatch) -> None:
    monkeypatch.setattr("typeless_local.app.capture_focus_context", lambda **_: FocusContext("TextEdit", "Untitled"))
    timers = []
    monkeypatch.setattr(app_module.threading, "Timer", lambda delay, fn: SimpleNamespace(
        daemon=False, start=lambda: timers.append(fn), cancel=lambda: None))
    app, hotkeys, clock = _hotkey_app(monkeypatch)

    hotkeys.last_primary_down_at = 10.0
    app._on_hotkey("hold_start")
    assert app.state == "recording" and app._holding
    assert app.audio_ducker.calls == []  # a slow Cmd-click must not blip the music
    for fn in timers:
        fn()
    assert app.audio_ducker.calls == ["duck"]
    app._on_hotkey("hold_end")

    assert app.state == "processing"


def test_a_slow_chord_only_drops_the_recording_its_hold_started(monkeypatch) -> None:
    monkeypatch.setattr("typeless_local.app.capture_focus_context", lambda **_: FocusContext("TextEdit", "Untitled"))
    app, hotkeys, clock = _hotkey_app(monkeypatch)

    app._on_hotkey("hands_free")  # a locked recording is running
    app._on_hotkey("hold_start")  # right Cmd held for a Cmd-click
    app._on_hotkey("hold_abort")
    app._on_hotkey("hold_end")
    assert app.state == "recording" and app.mode == "hands_free"

    app._on_hotkey("hands_free")  # stopped: now processing
    assert app.state == "processing"
    app._on_hotkey("hold_start")
    app._on_hotkey("hold_abort")
    assert app.state == "processing", "the dictation being transcribed is kept"

    app.state = "idle"
    app._on_hotkey("hold_start")
    app._on_hotkey("hold_abort")
    assert app.state == "idle"
    assert app.overlay.shown()[-1] == "cancelled"


def _deliver_into_claude(monkeypatch, where, lengths):
    pasted, copied = [], []
    monkeypatch.setattr(app_module, "paste_text", pasted.append)
    monkeypatch.setattr(app_module, "set_clipboard_text", copied.append)
    monkeypatch.setattr(app_module, "prepare_paste", lambda context: where)
    monkeypatch.setattr(app_module, "focused_text_length", lambda pid: lengths.pop(0) if lengths else 0)
    monkeypatch.setattr(app_module, "PASTE_CHECK_S", 0.0)
    app = _make_app("hello there")
    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("Claude", "", can_insert_text=True, pid=TARGET_PID))
    return app, pasted, copied


def test_a_paste_that_changed_nothing_does_not_claim_inserted(monkeypatch) -> None:
    app, pasted, _ = _deliver_into_claude(monkeypatch, "ok", [0, 0])

    assert pasted == ["Refined text."]
    assert app.overlay.shown()[-1] == "inserted-unsure"


def test_a_paste_that_grew_the_field_says_inserted(monkeypatch) -> None:
    app, pasted, _ = _deliver_into_claude(monkeypatch, "ok", [0, 12])

    assert pasted == ["Refined text."]
    assert app.overlay.shown()[-1] == "inserted"


@pytest.mark.parametrize("where", ["elsewhere", "lost"])
def test_no_blind_paste_when_the_text_field_is_gone(monkeypatch, where) -> None:
    app, pasted, copied = _deliver_into_claude(monkeypatch, where, [])

    assert pasted == []
    assert copied == ["Refined text."]
    assert app.overlay.shown()[-1] == "edit-notarget"


def test_an_app_with_no_readable_field_is_still_pasted_into(monkeypatch) -> None:
    app, pasted, _ = _deliver_into_claude(monkeypatch, "blind", [0, 0])

    assert pasted == ["Refined text."]
    assert app.overlay.shown()[-1] == "inserted"


def _audio(seconds: float) -> np.ndarray:
    return np.full(int(16000 * seconds), 0.5, dtype=np.float32)


def _auto_app(monkeypatch, system=("en", "zh")) -> TypelessLocalApp:
    monkeypatch.setattr(app_module.languages, "_system", frozenset(system))
    app = _make_app("hi")
    app.prefs = Preferences()
    app.asr._asr_config = {"language": ""}
    app.asr.probs = {"is": 0.5, "zh": 0.3, "en": 0.2}
    return app


def _detecting(app, probs) -> list:
    detections = []
    app.asr.detect_language = lambda audio: (detections.append(1), probs)[1]
    return detections


def test_a_short_clip_heard_in_an_allowed_language_costs_no_detection(monkeypatch) -> None:
    app = _auto_app(monkeypatch)
    app.asr.language = "zh"
    detections = _detecting(app, {"is": 0.5})

    app._whisper(_audio(2))

    assert detections == []
    assert app.asr.languages == [None]


def test_a_short_clip_heard_in_a_foreign_language_is_heard_again_in_the_likeliest_allowed_one(monkeypatch) -> None:
    app = _auto_app(monkeypatch)
    app.asr.language = "is"
    detections = _detecting(app, {"is": 0.5, "zh": 0.3, "en": 0.2})
    app.whisper_prompt = "Common terms: Jarvis."

    app._whisper(_audio(2))

    assert len(detections) == 1
    assert app.asr.languages == [None, "zh"]
    assert app.asr.prompts == ["Common terms: Jarvis.", f"{app_module.SIMPLIFIED_HINT} Common terms: Jarvis."]


def test_an_english_redo_gets_no_simplified_hint(monkeypatch) -> None:
    app = _auto_app(monkeypatch, system=("en",))
    app.asr.language = "ko"

    app._whisper(_audio(2))

    assert app.asr.languages == [None, "en"]
    assert app.asr.prompts == [None, None]


def test_short_clip_routing_leaves_long_clips_and_chosen_languages_alone(monkeypatch) -> None:
    app = _auto_app(monkeypatch)
    app.asr.language = "is"
    detections = _detecting(app, {"en": 1.0})
    app._whisper(_audio(4))  # not shorter than SHORT_CLIP_S
    app._whisper(_audio(35))
    for spoken in ("zh", "en"):
        app.asr._asr_config = {"language": spoken}
        app._whisper(_audio(2))

    assert detections == []
    assert app.asr.languages == [None] * 4


def test_a_failed_or_empty_detection_keeps_the_free_result(monkeypatch) -> None:
    app = _auto_app(monkeypatch)
    app.asr.language = "is"
    app.asr.detect_language = lambda audio: 1 / 0
    app._whisper(_audio(2))
    app.asr.detect_language = lambda audio: {"is": 1.0}  # nothing allowed has any probability
    app._whisper(_audio(2))

    assert app.asr.languages == [None, None]


def test_a_confident_long_dictation_teaches_its_language(monkeypatch) -> None:
    app = _auto_app(monkeypatch)
    saved = []
    app.set_preference = lambda key, value: (saved.append((key, value)), setattr(app, "prefs", dataclasses.replace(app.prefs, **{key: value})))
    app.asr = _FakeASR("Das ist ein langer Satz", language="de", confidence=0.9)
    app.asr._asr_config = {"language": ""}

    app._whisper(_audio(5))
    app._whisper(_audio(5))  # already known: not saved again

    assert saved == [("heard_languages", ["de"])]
    assert app._allowed_languages() == {"en", "zh", "de"}


def test_learning_needs_a_long_confident_automatic_transcript_of_a_known_language(monkeypatch) -> None:
    app = _auto_app(monkeypatch)
    saved = []
    app.set_preference = lambda key, value: saved.append(value)
    long_text = "Das ist ein langer Satz"
    for text, language, confidence, seconds, spoken in [
        (long_text, "de", 0.9, 2, ""),  # short clip
        (long_text, "de", 0.3, 5, ""),  # unsure
        ("Ja.", "de", 0.9, 5, ""),  # too little text
        (long_text, "de", 0.9, 5, "en"),  # language chosen in Settings
        (long_text, "xx", 0.9, 5, ""),  # not a Whisper language
    ]:
        app.asr = _FakeASR(text, language=language, confidence=confidence)
        app.asr._asr_config = {"language": spoken}
        app._whisper(_audio(seconds))

    assert saved == []


def test_heard_languages_are_capped(monkeypatch) -> None:
    app = _auto_app(monkeypatch)
    app.prefs = Preferences(heard_languages=["de", "fr", "es", "it", "pt", "nl", "pl", "ru"])
    saved = []
    app.set_preference = lambda key, value: saved.append(value)
    app.asr = _FakeASR("これは長い文章です", language="ja", confidence=0.9)
    app.asr._asr_config = {"language": ""}

    app._whisper(_audio(5))

    assert saved == [["fr", "es", "it", "pt", "nl", "pl", "ru", "ja"]]


def test_short_fragments_are_kept_or_dropped_by_the_allowed_languages(monkeypatch) -> None:
    app = _auto_app(monkeypatch)
    app.prefs = Preferences(heard_languages=["de"])

    def drops(text, language, confidence):
        return app._should_drop_transcript(Transcript(text, language, confidence))

    assert not drops("好的", "zh", 0.1)  # Chinese fragments were always kept
    assert not drops("OK.", "en", 0.8)
    assert not drops("Ja.", "de", 0.8)  # learned from earlier dictations
    assert drops("you", "en", 0.2)  # allowed but unsure
    assert drops("はい", "ja", 0.99)  # not a language of this user, however sure
    assert drops("Bye", "is", 0.9)
    assert not drops("A longer sentence in Icelandic", "is", 0.9)  # the filter is for short fragments
    app.asr._asr_config = {"language": "ja"}
    assert not drops("はい", "ja", 0.9)  # chosen in Settings


def test_a_hand_fix_before_sending_is_learned_into_the_vocabulary(tmp_path) -> None:
    app = _make_app("x")
    app.config.user_paths = SimpleNamespace(vocab_path=tmp_path / "v.yaml", trace_db_path=tmp_path / "t.db")
    reloads = []
    app.reload_vocab = lambda: reloads.append(1)
    app._store_sent_text(1, "还没有默制是吧？", "还没有merge是吧？")
    from typeless_local import vocab

    assert vocab.load_user_terms(tmp_path / "v.yaml") == ["merge"] and reloads == [1]
    app._store_sent_text(1, "还没有默制是吧？", "还没有merge是吧？")  # a repeat fix counts again, the word is not repeated
    assert vocab.load_user_terms(tmp_path / "v.yaml") == ["merge"] and reloads == [1, 1]


def test_english_practice_pastes_at_the_marker_once_then_shows_the_card(monkeypatch) -> None:
    pasted = []
    monkeypatch.setattr("typeless_local.app.paste_text", pasted.append)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("我想明天下午和你开个会讨论一下这个项目的进度")
    app.asr = _FakeASR("我想明天下午和你开个会讨论一下这个项目的进度", language="zh")
    app.prefs = Preferences(english_practice=True)
    card = {"en": "Hello, world.", "phrases": [{"en": "Hello", "zh": "你好"}]}

    def refine(text, context, vocab=None, language="", english=False, on_text=None):
        assert english and not pasted
        on_text("你好，世界。")
        assert pasted == ["你好，世界。"]  # pasted before refine returned
        return RefineResult(text="你好，世界。", raw_text=text, model="m", english=card)

    app.refiner.refine = refine

    app._process_audio(np.ones(16000, dtype=np.float32), FocusContext("Slack", "#general", can_insert_text=True))

    assert pasted == ["你好，世界。"]
    assert app.overlay.shown() == ["refining", "inserted", "english"]
    assert app.capsule.data["en"] == "Hello, world." and app.capsule.data["use"] is False


def test_clicking_words_on_the_english_card_saves_and_unsaves_them(tmp_path) -> None:
    from typeless_local import history

    app = _make_app("x")
    app.trace = SimpleNamespace(db_path=tmp_path / "trace.db")
    app.prefs = Preferences()
    app.windows = None
    app._english_card = {"en": "A meeting.", "source": "一个会议", "zh": {"set up": "安排"}}
    looked = []
    app.refiner = SimpleNamespace(define=lambda term, sentence, source="": looked.append((term, sentence)) or "会议")

    app._on_overlay_action("word", {"term": "meeting", "kind": "word", "on": True})
    app._on_overlay_action("word", {"term": "set up", "kind": "phrase", "on": True})
    for _ in range(100):
        if looked and history.list_words(app.trace.db_path)[-1]["meaning"]:
            break
        time.sleep(0.01)
    words = {w["term"]: w for w in history.list_words(app.trace.db_path)}
    assert words["meeting"]["meaning"] == "会议" and looked == [("meeting", "A meeting.")]
    assert words["set up"]["meaning"] == "安排" and words["set up"]["source"] == "一个会议"

    app._on_overlay_action("word", {"term": "meeting", "kind": "word", "on": False})
    assert [w["term"] for w in history.list_words(app.trace.db_path)] == ["set up"]


def test_dragged_phrase_without_zh_looks_up_its_meaning(tmp_path) -> None:
    from typeless_local import history

    app = _make_app("x")
    app.trace = SimpleNamespace(db_path=tmp_path / "trace.db")
    app.prefs = Preferences()
    app.windows = None
    app._english_card = {"en": "Let's set up a meeting.", "source": "", "zh": {"set up": "安排"}}
    looked = []
    app.refiner = SimpleNamespace(define=lambda term, sentence, source="": looked.append(term) or "安排会议")

    app._on_overlay_action("word", {"term": "set up a meeting", "kind": "phrase", "on": True})
    for _ in range(100):
        if history.list_words(app.trace.db_path)[-1]["meaning"]:
            break
        time.sleep(0.01)
    assert looked == ["set up a meeting"]
    assert history.list_words(app.trace.db_path)[-1]["meaning"] == "安排会议"

    app._on_overlay_action("word", {"term": "set up a meeting", "kind": "phrase", "on": False})
    assert history.list_words(app.trace.db_path) == []


class _InstantTimer:
    def __init__(self, delay, fn, args=()):
        self.fn, self.args = fn, args

    def start(self):
        self.fn(*self.args)


def _send_app(monkeypatch):
    returns = []
    monkeypatch.setattr(app_module, "press_return", lambda: returns.append("return"))
    monkeypatch.setattr(app_module.threading, "Timer", _InstantTimer)
    monkeypatch.setattr(app_module, "paste_text", lambda text: None)
    monkeypatch.setattr(app_module, "set_clipboard_text", lambda text: None)
    monkeypatch.setattr(app_module, "prepare_paste", lambda context: "blind")
    app = _make_app("hello")
    app._active_session_id = 3
    app._send_session = 3
    return app, returns


def _deliver(app, fallback=""):
    context = FocusContext("TextEdit", "", can_insert_text=True, pid=TARGET_PID)
    app._deliver("Text.", "raw", context, fallback, SimpleNamespace(was_pasted=False))


def test_enter_finish_presses_return_once_after_the_paste(monkeypatch) -> None:
    app, returns = _send_app(monkeypatch)
    _deliver(app)
    assert returns == ["return"]
    _deliver(app)
    assert returns == ["return"], "the flag is used up by its dictation"


@pytest.mark.parametrize("fallback", ["net", "key"])
def test_no_return_after_a_raw_fallback_paste(monkeypatch, fallback) -> None:
    app, returns = _send_app(monkeypatch)
    _deliver(app, fallback)
    assert returns == []


def test_no_return_when_the_paste_did_not_land(monkeypatch) -> None:
    app, returns = _send_app(monkeypatch)
    monkeypatch.setattr(app_module, "prepare_paste", lambda context: "lost")
    _deliver(app)
    assert returns == []


def test_no_return_after_a_new_dictation_began(monkeypatch) -> None:
    app, returns = _send_app(monkeypatch)
    app._active_session_id = 4
    _deliver(app)
    assert returns == []


def test_send_action_marks_the_dictation_and_recording_start_clears_it(monkeypatch) -> None:
    app = _recording_app()
    app.state = "recording"
    app._active_session_id = 5
    finished = []
    monkeypatch.setattr(app, "_finish_recording", lambda: finished.append(1))
    app._on_hotkey("send")
    assert app._send_session == 5 and finished == [1]

    app.state = "idle"
    app._on_hotkey("send")
    assert finished == [1], "Enter outside a recording does nothing"

    app._start_recording("tap")
    assert app._send_session is None


class _FakeCloud:
    model = "qwen-audio-3.1-asr-flash-streaming"

    def __init__(self, transcript: Transcript | None) -> None:
        self.transcript = transcript
        self.cancelled = False

    def finish(self, audio):
        return self.transcript

    def cancel(self) -> None:
        self.cancelled = True


def test_qwen_heard_the_dictation_so_whisper_is_not_asked(monkeypatch) -> None:
    monkeypatch.setattr("typeless_local.app.paste_text", lambda _text: None)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("whisper words")
    app._qwen_heard = deque()
    cloud = _FakeCloud(Transcript(text="千问听到的。", language="zh", confidence=1.0))
    context = FocusContext(app_name="TextEdit", window_title="Untitled", can_insert_text=True)

    app._process_audio(np.ones(16000, dtype=np.float32), context, None, (), 0, cloud)

    assert app.asr.calls == []
    assert app.refiner.calls == [("千问听到的。", context)]
    assert cloud.cancelled
    assert [(heard.text, heard.app) for heard in app._qwen_heard] == [("千问听到的。", "TextEdit")]


def test_qwen_reads_what_it_heard_here_lately_then_the_window_and_the_text_before() -> None:
    app = _make_app("")
    app.prefs = Preferences(send_before_text=True)
    now = time.time()
    app._qwen_heard = deque([
        Recent(text="一小时前", app="Claude", at=now - 3600),
        Recent(text="别的 App 里", app="Ghostty", at=now - 10),
        Recent(text="一", app="Claude", at=now - 50),
        Recent(text="二", app="Claude", at=now - 40),
        Recent(text="三 Jev", app="Claude", at=now - 30),
        Recent(text="已在框里", app="Claude", at=now - 20),
    ])
    context = FocusContext(app_name="Claude", window_title="Jev 工具路由", before_text="前面写着已在框里")

    assert app._qwen_context(context) == ["一", "二", "三 Jev", "Jev 工具路由", "前面写着已在框里"]

    app.prefs = Preferences(send_window_title=False, send_before_text=False)
    assert app._qwen_context(context) == ["二", "三 Jev", "已在框里"]


def test_qwen_hears_the_chosen_language_or_all_of_his(monkeypatch) -> None:
    app = _make_app("")
    app.prefs = Preferences(heard_languages=["zh"])
    monkeypatch.setattr(app_module.languages, "known", lambda: False)
    assert app._qwen_languages() == [], "English alone before the Mac's languages are read would garble Chinese"

    monkeypatch.setattr(app_module.languages, "known", lambda: True)
    monkeypatch.setattr(app_module.languages, "allowed", lambda heard: {"zh", "en"})
    assert app._qwen_languages() == ["en", "zh"]

    app.asr._asr_config = {"language": "fr"}
    assert app._qwen_languages() == ["fr"]


def test_whisper_hears_the_whole_recording_when_qwen_fails(monkeypatch) -> None:
    monkeypatch.setattr("typeless_local.app.paste_text", lambda _text: None)
    monkeypatch.setattr("typeless_local.app.set_clipboard_text", lambda text: None)
    app = _make_app("whisper words")
    context = FocusContext(app_name="TextEdit", window_title="Untitled", can_insert_text=True)

    app._process_audio(np.ones(16000, dtype=np.float32), context, None, (), 0, _FakeCloud(None))

    (heard,) = app.asr.calls
    assert heard.size == 16000
    assert app.refiner.calls == [("whisper words", context)]


def _finish_with_a_tap_then_tap_again(monkeypatch, app, hotkeys, clock, second_at: float) -> None:
    clock[0] = hotkeys.last_primary_down_at = 40.0
    app._on_hotkey("primary_down")
    clock[0] = hotkeys.last_primary_up_at = 40.1
    app._on_hotkey("primary_up")
    clock[0] = hotkeys.last_primary_down_at = 45.0
    app._on_hotkey("primary_down")  # finishes the dictation
    clock[0] = hotkeys.last_primary_up_at = 45.08
    app._on_hotkey("primary_up")
    clock[0] = hotkeys.last_primary_down_at = second_at
    app._on_hotkey("primary_down")


def test_double_tap_to_finish_sends_once_pasted_and_starts_nothing(monkeypatch) -> None:
    app, hotkeys, clock = _hotkey_app(monkeypatch)
    _finish_with_a_tap_then_tap_again(monkeypatch, app, hotkeys, clock, 45.25)

    assert app.state == "processing"
    assert len(app.executor.submissions) == 1
    assert app._send_session == app._active_session_id


def test_a_late_second_tap_does_not_send(monkeypatch) -> None:
    app, hotkeys, clock = _hotkey_app(monkeypatch)
    _finish_with_a_tap_then_tap_again(monkeypatch, app, hotkeys, clock, 46.0)

    assert app._send_session is None
    assert len(app.executor.submissions) == 1


def test_second_tap_after_the_paste_sends_right_away(monkeypatch) -> None:
    app, returns = _send_app(monkeypatch)
    app._send_session = None
    _deliver(app)
    assert returns == []

    app._send_when_pasted(3)
    assert returns == ["return"]


def test_second_tap_after_a_raw_fallback_paste_does_not_send(monkeypatch) -> None:
    app, returns = _send_app(monkeypatch)
    app._send_session = None
    _deliver(app, "net")
    app._send_when_pasted(3)
    assert returns == []
