"""The app's coordinator: hotkey, microphone, recognizer, refinement, the
capsule, and the text going into the focused app."""

from __future__ import annotations

import atexit
from collections import deque
import dataclasses
import json
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unicodedata
from typing import Callable, Literal

from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
import numpy as np
from PyObjCTools import AppHelper

from typeless_local import app_version, brand, cloud_asr, diagnostics, feedback, i18n, keyboard_layout, keychain, languages, permissions, reach, trial, usage
from typeless_local.i18n import t
from typeless_local.asr import JarvisASR, Transcript, mlx_whisper_repo
from typeless_local.audio import MicrophoneRecorder, keep_recording, peak_level
from typeless_local import devices
from typeless_local.capsule import INSERTED_STATES, Capsule
from typeless_local.config import (
    AppConfig,
    adopt_default_preset,
    load_config,
    migrate_legacy_config_dir,
    preset_names,
    refine_config_for,
    save_input_device,
    save_default_preset,
    save_user_setting,
)
from typeless_local.mac_integration import (
    FocusContext,
    GlobalHotkeyMonitor,
    capture_focus_context,
    caret_rect,
    focused_text_value,
    frontmost_pid,
    has_accessibility_trust,
    paste_text,
    prepare_paste,
    press_return,
    focused_text_length,
    press_play_pause,
    request_accessibility_trust,
    set_clipboard_text,
    undo_last_edit,
)
from typeless_local.first_run import (
    NotEnoughDisk,
    _store_api_key,
    download_model,
    ensure_api_key,
    model_is_cached,
    set_api_key,
)
from typeless_local import history as word_book
from typeless_local.history import purge_older_than, set_sent_text, term_counts
from typeless_local.sent_text import SentTextWatcher
from typeless_local.menubar import MenuBarIcon, Recent, Snapshot
from typeless_local.overlay import FloatingOverlay
from typeless_local.preferences import Preferences, load_preferences, save_preference
from typeless_local.refine import MissingAPIKey, RefineResult, TextRefiner, TrialUnavailable
from typeless_local.stats import DailyStats
from typeless_local.trace import DictationTrace, SessionRecord, append_correction
from typeless_local.vocab import as_initial_prompt, hotword_weights, learn_term, load_user_terms, load_vocab, whisper_terms, write_starter_file
from typeless_local.windows import Windows, install_main_menu, learned_fixes, prices

LOGGER = logging.getLogger(__name__)
Mode = Literal["tap", "hands_free"]
USE_DAYS = 90  # how far back word use counts toward Whisper's prompt
DOUBLE_CLICK_SECONDS = 0.4
# After pressing play/pause, how long to watch for media that started instead of stopping.
MEDIA_CHECK_S = 0.8
MAX_UNCLEAR_MEDIA_APPS = 8  # players remembered as unconfirmable; keeps the saved list small
MEDIA_POLL_S = 0.05
LONG_PRESS_SECONDS = 0.6
# A right-Cmd hold this long into its recording is talking, not a slow chord:
# only then are the speakers muted and the music paused.
HOLD_CONFIRM_S = 0.6
# How long a pasted dictation gets to show up in the field before it counts as lost.
PASTE_CHECK_S = 0.5
# After a paste, the wait before the Return that sends it: the paste lands first.
SEND_SETTLE_S = 0.1
# The second tap of a double tap that finished a dictation and the paste can
# race; deciding to send is done under this lock on both sides.
_SEND_LOCK = threading.Lock()
MIN_MIC_STARTUP_SECONDS = 0.75
# A non-Chinese transcript this short is usually Whisper inventing a word over
# noise ("you", "Bye."), but it is also how "OK" and "Yes" come out. Whisper's
# own confidence tells them apart: invented words score low. A language the
# user does not speak (not in languages.allowed) is dropped whatever the score.
SHORT_FRAGMENT_CHARS = 5
WHISPER_WINDOW_S = 30
# Whisper's prompt is text it takes to have come before: Simplified text
# here keeps a forced-Chinese transcript Simplified.
SIMPLIFIED_HINT = "以下是普通话的简体中文转录。"
PREWARM_INTERVAL_S = 3.0
MIN_SHORT_ENGLISH_CONFIDENCE = 0.4
# After a card gives the keyboard back, before Cmd+Z is sent: the undo has to
# reach the app the text went to, not the card.
KEY_HANDBACK_S = 0.25
# How often to look again for the Accessibility permission while it is missing.
HOTKEY_RETRY_S = 2.0
STATS_FIRST_S = 60.0
STATS_EVERY_S = 6 * 3600.0
# What the capsule's 延长 button adds to a recording nearing its limit.
EXTEND_RECORDING_S = 15 * 60.0
# Between taking the old text back and pasting the new one.
UNDO_SETTLE_S = 0.15
# Qwen's context: what it heard in the same app this recently, up to this many dictations.
QWEN_CONTEXT_S = 30 * 60.0
QWEN_CONTEXT_DICTATIONS = 3
_DEFAULT_PREFERENCES = Preferences()
# What the capsule says went wrong, by the step that failed.
_FAILED_STEP = {
    "asr": ("转写失败", "Transcription failed"),
    "refine": ("润色失败", "Refinement failed"),
    "paste": ("粘贴失败", "Paste failed"),
}


@dataclasses.dataclass(frozen=True)
class Insertion:
    """A paste the capsule can still undo or replace.

    Only until the user types: after that, Cmd+Z would take back their own
    typing instead, so any key press forgets it.
    """

    text: str
    raw: str
    pid: int
    context: FocusContext


class TypelessLocalApp:
    """Coordinate hotkeys, recording, ASR, refinement, UI, and insertion."""

    def __init__(self, config: AppConfig, headless: bool = False) -> None:
        self.config = config
        self.headless = headless

        # Vocab + trace are file-backed and cheap; load them before components
        # so _build_components-injected fakes see a fully initialized app shell.
        user_paths = getattr(config, "user_paths", None)
        if user_paths is not None:
            write_starter_file(user_paths.vocab_path)
            self.vocab = load_vocab(user_paths.vocab_path)
            self.whisper_prompt = self._whisper_prompt(user_paths)
            self.hotword_weights = hotword_weights(user_paths.vocab_path)
            self.trace = DictationTrace(user_paths.trace_db_path)
            self.daily_stats = DailyStats(user_paths.config_dir / "stats.json", app_version())
        else:
            self.daily_stats = None
            self.vocab = []
            self.whisper_prompt = ""
            self.hotword_weights = {}
            self.trace = None
        self.prefs = load_preferences(user_paths)
        self._trial_over = self.prefs.trial_used_up
        if dataclasses.is_dataclass(config):
            self.config = dataclasses.replace(config, max_recording_seconds=self.prefs.max_minutes * 60.0)

        if not headless:
            # Before anything with words in it is built.
            i18n.use(self.prefs.ui_language)

        components = self._build_components()
        self.asr = components.asr
        self.refiner = components.refiner
        self.recorder = components.recorder

        if not headless:
            self.overlay = FloatingOverlay.alloc().init()
            self.overlay.set_action_callback(self._on_overlay_action)
            self.overlay.set_hover_callback(self._on_overlay_hover)
            try:
                from typeless_local._vendor.jarvis_core.media_ducking import SystemAudioDucker
            except ImportError:
                from core.media_ducking import SystemAudioDucker

            self.audio_ducker = SystemAudioDucker.from_config(config.jarvis_config)
            atexit.register(self.audio_ducker.restore_all)
            self.hotkeys = GlobalHotkeyMonitor(
                self._on_hotkey,
                debug_hotkey=config.debug_hotkey,
                is_active_fn=lambda: self.state != "idle",
                is_recording_fn=lambda: self.state == "recording",
                watch_keys_fn=lambda: self._keys_wanted,
                use_f5_fn=lambda: self.prefs.f5_hotkey,
                use_right_command_fn=lambda: self.prefs.right_command_hotkey,
            )

            self.menubar = MenuBarIcon(on_action=self._on_menu_action, snapshot=self._menu_snapshot)
            self.windows = Windows(self)
            self.sent_watcher = SentTextWatcher(focused_text_value, frontmost_pid, self._store_sent_text)
        else:
            self.sent_watcher = None
            self.overlay = None
            self.audio_ducker = None
            self.hotkeys = None
            self.menubar = None
            self.windows = None
        self.capsule = Capsule(self.overlay, self._call_ui)
        self.capsule.inserted_dismiss_s = self.prefs.dismiss_seconds

        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="typeless-local")
        # Muting and unmuting shell out to osascript, which is too slow to run
        # between the hotkey and the microphone opening. One worker keeps every
        # restore behind the duck it undoes.
        self._audio_io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="typeless-audio-io")
        self.state = "idle"
        self._set_menubar("idle")
        self.mode: Mode = "tap"
        self.focus_context = FocusContext(app_name="", window_title="", selected_text="")
        self._copy_fallback_text = ""
        self._lock = threading.RLock()
        self._processing_started_at = 0.0
        self._recording_timer: threading.Timer | None = None
        self._finish_debounce_timer: threading.Timer | None = None
        self._hold_timer: threading.Timer | None = None
        self._holding = False
        self._recording_started_at = 0.0
        self._primary_down_at = 0.0
        self._last_short_tap_at = 0.0
        self._active_session_id = 0
        self._stretches: list[Future[Transcript]] = []
        # This dictation's stream to Qwen when that is the recognizer; None means Whisper.
        self._cloud: cloud_asr.QwenStream | None = None
        # What Qwen heard lately, and in which app: context for the next dictation there.
        self._qwen_heard: deque[Recent] = deque(maxlen=20)
        self._capture_device = ""
        self._ducked = False
        self._insertion: Insertion | None = None
        self._recent: Recent | None = None
        # Read inside the keyboard event tap on every key press: a plain attribute.
        self._keys_wanted = False
        self._download: tuple[float, float] | None = None

    @property
    def prefs(self) -> Preferences:
        return self.__dict__.get("_prefs") or _DEFAULT_PREFERENCES

    @prefs.setter
    def prefs(self, value: Preferences) -> None:
        self.__dict__["_prefs"] = value

    def _build_components(self) -> SimpleNamespace:
        """Construct ASR / refiner / recorder. Patched by tests to inject fakes."""

        config = self.config
        asr = JarvisASR(config.jarvis_root, config.jarvis_config)
        refiner = TextRefiner(config.refine)
        recorder = MicrophoneRecorder(
            sample_rate=config.sample_rate,
            on_level=self._on_audio_level,
            device=devices.resolve_input_index(getattr(config, "input_device", "")),
            on_stretch=self._on_stretch,
            input_channel=getattr(config, "input_channel", 0),
        )
        return SimpleNamespace(asr=asr, refiner=refiner, recorder=recorder)

    def reload_vocab(self) -> None:
        """Atomically replace ``self.vocab`` with a freshly loaded list."""

        user_paths = getattr(self.config, "user_paths", None)
        if user_paths is None:
            return
        self.vocab = load_vocab(user_paths.vocab_path)
        self.whisper_prompt = self._whisper_prompt(user_paths)
        self.hotword_weights = hotword_weights(user_paths.vocab_path)
        LOGGER.info("Reloaded vocab: %d terms", len(self.vocab))

    @staticmethod
    def _whisper_prompt(user_paths) -> str:
        """Whisper's prompt from the hottest words; counted on launch and on vocabulary changes only."""

        terms = load_user_terms(user_paths.vocab_path)
        uses = term_counts(user_paths.trace_db_path, terms, days=USE_DAYS) if user_paths.trace_db_path.exists() else {}
        return as_initial_prompt(whisper_terms(user_paths.vocab_path, uses))

    def _select_capture_device(self) -> str:
        """Point the recorder at the preferred mic, or the system default if it is gone.

        Resolved per recording rather than once at launch: the mic gets plugged
        and pulled between dictations, and the answer has to be the device that
        is actually there now. Returns the name being captured with, which is
        what decides whether the speakers still need ducking.
        """

        if not getattr(getattr(self, "recorder", None), "wedged", False):
            devices.refresh_if_changed()  # a refresh while a stream is stuck hangs too
        return self._resolve_capture_device()

    def _resolve_capture_device(self) -> str:
        preferred = getattr(self.config, "input_device", "")
        index = devices.resolve_input_index(preferred)
        capture = preferred
        if index is None:
            capture = devices.current_input_device()
            # A Bluetooth headset's mic puts the headset in call mode while it is
            # open: the music cuts out and comes back at phone quality. The Mac's
            # own mic hears just as well at the desk, and with the sound in the
            # headset it has no echo to pick up. A headset chosen by name is kept.
            built_in = devices.built_in_input() if devices.default_input_is_bluetooth() else ""
            built_in_index = devices.resolve_input_index(built_in) if built_in else None
            if built_in_index is not None:
                if built_in != getattr(self, "_capture_device", ""):
                    LOGGER.info("Recording with %s, not %s: a Bluetooth mic drops the headset to call quality", built_in, capture)
                index, capture = built_in_index, built_in
        recorder = getattr(self, "recorder", None)
        if recorder is not None:
            recorder.device = index
        return capture

    def _speakers_need_ducking(self, capture: str) -> bool:
        """Whether the speakers must be silenced for this capture to stay clean.

        Headphones need nothing: what they play never reaches the mic. A capture
        device with hardware echo cancellation already removes what the
        speakers play, but only while it is fed the same signal they get, so the
        answer depends on the input and output pairing rather than on the mic
        alone. Anything unrecognised ducks, which is the behaviour that was there
        before any pairing was configured. ``capture`` is the device actually in
        use, not the configured preference, so falling back to the built-in mic
        ducks even while the paired mic stays configured.
        """

        playback = devices.current_output_device()
        if devices.output_is_headphones():
            LOGGER.info("Leaving system audio up: %s plays into headphones", playback)
            return False
        pairs = list(getattr(self.config, "aec_pairs", ()) or ())
        # The paired mic cancels the speakers only on its first channel.
        if not getattr(self.config, "input_channel", 0) and devices.has_hardware_aec(capture, playback, pairs):
            LOGGER.info(
                "Leaving system audio up: %s + %s cancels the speakers in hardware",
                capture,
                playback,
            )
            return False
        return True

    def select_input_device(self, name: str) -> None:
        """Point capture at ``name`` from the next recording on, and persist it."""

        self.config = dataclasses.replace(self.config, input_device=name)
        recorder = getattr(self, "recorder", None)
        if recorder is not None:
            recorder.device = devices.resolve_input_index(name)
        user_paths = getattr(self.config, "user_paths", None)
        if user_paths is not None:
            save_input_device(user_paths, name)
        self._windows_changed()
        LOGGER.info("Capture device set to %s", name or "(system default)")

    def select_model(self, preset: str) -> None:
        """Switch the refinement preset live and persist it to the user config."""

        refine = refine_config_for(self.config.jarvis_config, preset)
        self.config = dataclasses.replace(self.config, refine=refine)
        self.refiner = TextRefiner(refine)
        user_paths = getattr(self.config, "user_paths", None)
        if user_paths is not None:
            save_default_preset(user_paths, preset)
        self._refresh_issues()
        self._windows_changed()
        LOGGER.info("Refinement model switched to %s (%s)", preset, refine.model)

    def set_language(self, language: str) -> None:
        """Recognise ``language`` ("" detects it) from the next dictation on, and persist it."""

        language = language if language in ("zh", "en", "fr") else ""
        set_language = getattr(self.asr, "set_language", None)
        if set_language is not None:
            set_language(language)
        self._engine_setting("asr", "language", language)
        LOGGER.info("Recognition language set to %s", language or "auto")

    def set_ducking(self, enabled: bool) -> None:
        """Whether other apps' sound is lowered while recording."""

        ducker = getattr(self, "audio_ducker", None)
        if ducker is not None:
            ducker.enabled = bool(enabled)
        self._engine_setting("audio_ducking", "enabled", bool(enabled))

    def _engine_setting(self, section: str, key: str, value) -> None:
        """A setting the engine reads from its own config section: live and on disk."""

        jarvis = getattr(self.config, "jarvis_config", None)
        if isinstance(jarvis, dict):
            if not isinstance(jarvis.get(section), dict):
                jarvis[section] = {}
            jarvis[section][key] = value
        user_paths = getattr(self.config, "user_paths", None)
        if user_paths is not None:
            save_user_setting(user_paths, section, key, value)

    def store_api_key(self, env_name: str, key: str) -> None:
        """Keep ``key`` for ``env_name`` (keychain first) and use it from the next dictation."""

        user_paths = getattr(self.config, "user_paths", None)
        env_path = user_paths.env_path if user_paths is not None else Path(os.devnull)
        _store_api_key(env_name, key, env_path)
        if env_name == trial.OWN_KEY_ENV and self.config.refine.preset == trial.PRESET:
            # Their own key replaces the free trial rather than sitting unused beside it.
            self._trial_over = False
            refine = refine_config_for(self.config.jarvis_config, trial.OWN_KEY_PRESET)
            self.config = dataclasses.replace(self.config, refine=refine)
            self.refiner = TextRefiner(refine)
            if user_paths is not None:
                save_default_preset(user_paths, trial.OWN_KEY_PRESET)
            self._windows_changed()
            LOGGER.info("Own OpenAI key saved; switched from the free trial to %s", trial.OWN_KEY_PRESET)
        elif env_name == self.config.refine.api_key_env:
            # The client holds the key it was made with.
            self.refiner = TextRefiner(self.config.refine)
        self._call_ui(self._refresh_issues)
        LOGGER.info("%s saved", env_name)

    def _tell_guide(self, message: dict) -> None:
        windows = getattr(self, "windows", None)
        if windows is not None:
            self._call_ui(windows.guide, message)

    def _windows_changed(self, history: bool = False) -> None:
        windows = getattr(self, "windows", None)
        if windows is not None:
            self._call_ui(windows.refresh, history)

    def set_preference(self, key: str, value) -> Preferences:
        """Change one setting from the Settings window: validate, persist, apply."""

        user_paths = getattr(self.config, "user_paths", None)
        self.prefs = save_preference(user_paths, self.prefs, key, value)
        self._apply_preferences(changed=key)
        return self.prefs

    def _apply_preferences(self, changed: str | None = None) -> None:
        """Make the running app follow the preferences (all of them, or one that changed)."""

        prefs = self.prefs
        capsule = getattr(self, "capsule", None)
        if capsule is not None:
            capsule.inserted_dismiss_s = prefs.dismiss_seconds
            if changed in (None, "show_handle"):
                capsule.set_handle(prefs.show_handle)
            if changed == "capsule_position" and prefs.capsule_position == "bottom":
                capsule.set_anchor("bottom")
        config = getattr(self, "config", None)
        if changed in (None, "max_minutes") and dataclasses.is_dataclass(config):
            self.config = dataclasses.replace(config, max_recording_seconds=prefs.max_minutes * 60.0)
        if changed in (None, "history_days"):
            self._purge_history()
        if changed in (None, "refine"):
            self._refresh_issues()
        if changed == "save_history":
            self._windows_changed(history=True)
        if changed == "ui_language":
            i18n.use(prefs.ui_language)
            self._relocalize()
        if changed in (None, "model_source"):
            reach.use_model_source(prefs.model_source)
        if changed == "send_usage_stats" and getattr(self, "daily_stats", None):
            if prefs.send_usage_stats:
                self.daily_stats.enabled = True
            else:
                self.daily_stats.forget()

    def _relocalize(self) -> None:
        """Redraw everything that has words in it, in the language just chosen."""

        def run() -> None:
            install_main_menu()
            if getattr(self, "menubar", None) is not None:
                self.menubar.set_state(self.state)
            if getattr(self, "overlay", None) is not None:
                self.overlay.relocalize()
            if getattr(self, "windows", None) is not None:
                self.windows.relocalize()

        self._call_ui(run)

    def _purge_history(self) -> None:
        """Drop dictations older than the limit the user picked; none by default."""

        days = self.prefs.history_days
        user_paths = getattr(getattr(self, "config", None), "user_paths", None)
        if days <= 0 or user_paths is None:
            return
        threading.Thread(
            target=purge_older_than, args=(user_paths.trace_db_path, days), daemon=True, name="history-purge"
        ).start()

    def _first_run(self) -> None:
        """Open the guide on first launch, unless everything it sets up is already there."""

        windows = getattr(self, "windows", None)
        if windows is None:
            self._prompt_for_missing_api_key()
            return
        if self.prefs.onboarding_done:
            return
        if self._set_up_already():
            # Someone who used the app before the guide existed: F5 is what they know.
            self.set_preference("onboarding_done", True)
            self.set_preference("f5_hotkey", True)
            return
        windows.show_onboarding()

    def _set_up_already(self) -> bool:
        asr_config = self.config.jarvis_config.get("asr") or {}
        provider = str(asr_config.get("provider") or "").strip().lower()
        return (
            bool(os.environ.get(self.config.refine.api_key_env or ""))
            and has_accessibility_trust()
            and permissions.microphone_status() == "authorized"
            and (provider != "mlx_whisper" or model_is_cached(mlx_whisper_repo(asr_config)))
        )

    def _prompt_for_missing_api_key(self) -> None:
        """Ask for the key on first launch, once the app can show a window."""

        user_paths = getattr(self.config, "user_paths", None)
        if user_paths is None:
            return
        ensure_api_key(
            self.config.refine.api_key_env,
            self.config.refine.model,
            user_paths.env_path,
        )

    def change_api_key(self) -> None:
        """Replace the key for the active preset from the menu bar."""

        user_paths = getattr(self.config, "user_paths", None)
        if user_paths is None:
            return
        if set_api_key(
            self.config.refine.api_key_env,
            self.config.refine.model,
            user_paths.env_path,
        ):
            LOGGER.info("%s updated", self.config.refine.api_key_env)
        self._refresh_issues()

    def open_settings(self, pane: str | None = None) -> None:
        """Bring up Settings at ``pane``."""

        windows = getattr(self, "windows", None)
        if windows is not None:
            self._call_ui(windows.show_settings, pane)
        elif pane == "model":
            self._call_ui(self.change_api_key)

    def open_history(self) -> None:
        windows = getattr(self, "windows", None)
        if windows is not None:
            self._call_ui(windows.show_history)

    # ------------------------------------------------------------ menu bar

    def _on_menu_action(self, key: str) -> None:
        """A menu-bar item was chosen (main thread)."""

        kind, _, arg = key.partition(":")
        if kind == "toggle":
            with self._lock:
                if self.state == "idle":
                    self._start_recording("tap")
                elif self.state == "recording":
                    self._finish_recording()
        elif kind == "latch":
            with self._lock:
                if self.state == "idle":
                    self._start_recording("hands_free")
        elif kind == "copy":
            self._copy_last_transcript()
        elif kind == "english":
            self.set_preference("english_practice", not self.prefs.english_practice)
        elif kind == "preset":
            self.select_model(arg)
        elif kind == "input":
            self.select_input_device(arg)
        elif kind == "settings":
            self.open_settings(arg or None)
        elif kind == "history":
            self.open_history()
        elif kind == "fix":
            self._fix_issue(arg)
        elif kind == "quit":
            NSApplication.sharedApplication().terminate_(None)

    def _restart(self) -> None:
        """Quit and open again: the one cure for a microphone CoreAudio hung on."""

        self._run_audio_io(getattr(self.audio_ducker, "restore_all", lambda: None))
        if getattr(sys, "frozen", False):
            bundle = Path(sys.executable).resolve().parents[2]
            # Opened once this process is gone, or the single-instance check refuses it.
            subprocess.Popen(["/bin/sh", "-c", f'sleep 1.5; /usr/bin/open "{bundle}"'], start_new_session=True)
        NSApplication.sharedApplication().terminate_(None)

    def _fix_issue(self, issue: str) -> None:
        if issue == "perm":
            request_accessibility_trust()
            permissions.open_url(permissions.ACCESSIBILITY_SETTINGS)
        elif issue == "mic":
            if permissions.microphone_status() == "not_determined":
                permissions.request_microphone()
            else:
                permissions.open_url(permissions.MICROPHONE_SETTINGS)
        elif issue in ("key", "trial"):
            self.open_settings("model")
        elif issue == "model" and getattr(self, "_download", None) is None:
            self._start_model_prefetch()

    def _current_issues(self) -> tuple[str, ...]:
        """What only the user can fix right now; the menu-bar icon wears a badge for it."""

        issues = []
        if not has_accessibility_trust():
            issues.append("perm")
        if permissions.microphone_status() in ("denied", "restricted"):
            issues.append("mic")
        if getattr(self, "_model_failed", ""):
            issues.append("model")
        refine = getattr(self.config, "refine", None)
        if self.prefs.refine and refine is not None and not os.environ.get(refine.api_key_env or ""):
            issues.append("key")
        elif self.prefs.refine and getattr(self, "_trial_over", False) and getattr(refine, "preset", "") == trial.PRESET:
            issues.append("trial")
        return tuple(issues)

    def _count_dictation(self, record) -> None:
        """Today's anonymous counts: one dictation, its length, and what the trial spent on it."""

        stats = getattr(self, "daily_stats", None)
        text = getattr(record, "refined_text", "") or ""
        if stats is None or not text or not self.prefs.send_usage_stats:
            return
        spend = 0.0
        refine = getattr(self.config, "refine", None)
        if getattr(refine, "preset", "") == trial.PRESET:
            spend = usage.cost(
                refine.model, record.prompt_tokens or 0, record.cached_tokens or 0, record.completion_tokens or 0
            ) or 0.0
        stats.record(dictations=1, chars=len(text), trial_spend=spend)

    def _worker_url(self, path: str) -> str:
        try:
            base = refine_config_for(self.config.jarvis_config, trial.PRESET).base_url
        except Exception:
            return ""
        server = trial.server(base)
        return f"{server}/{path}" if server.startswith("https://") and "YOUR-SUBDOMAIN" not in server else ""

    def _stats_url(self) -> str:
        return self._worker_url("stats")

    def send_feedback(self, message: str, email: str, attach: bool, done: Callable[[bool, str], None]) -> None:
        """Post the About pane's feedback to the Worker from a background thread; ``done(ok, error)`` reports back."""

        url = self._worker_url("feedback")
        user_paths = getattr(self.config, "user_paths", None)
        # Gathered here on the calling thread, like export_diagnostics does.
        summary = self._diagnostic_summary() if attach else None
        secrets = [os.environ.get(name, "") for name in api_key_names(self.config)]
        asr_config = (self.config.jarvis_config or {}).get("asr") or {}

        def run() -> None:
            try:
                if not url:
                    raise RuntimeError("no feedback server configured")
                extra = None
                if summary is not None and user_paths is not None:
                    extra = feedback.diagnostics_text(summary, user_paths.log_path, secrets)
                body = feedback.build(
                    message, email, version=app_version(), spoken_language=str(asr_config.get("language") or ""),
                    ui_language=i18n.current(), extra=extra,
                )
                feedback.post(url, body)
                done(True, "")
            except Exception as exc:
                LOGGER.info("Feedback not sent (%s)", exc)
                done(False, t("没有发出去，请检查网络后重试。", "Could not send; check your connection and try again."))

        threading.Thread(target=run, daemon=True, name="feedback").start()

    def _send_stats(self) -> None:
        """Once a day: the finished days' counts, from a background thread. Reschedules itself."""

        AppHelper.callLater(STATS_EVERY_S, self._send_stats)
        stats, url = getattr(self, "daily_stats", None), self._stats_url()
        # Nothing goes out before the guide, which is where the user is told about it.
        if stats is None or not url or not self.prefs.send_usage_stats or not self.prefs.onboarding_done:
            return

        def run() -> None:
            stats.record()  # the app ran today, even with no dictation
            stats.send(url)

        threading.Thread(target=run, daemon=True, name="usage-stats").start()

    def _fallback_for(self, exc: BaseException) -> str:
        """Why the raw transcript went in: "trial", "key", "timeout" or "error"."""

        if isinstance(exc, TrialUnavailable):
            self._trial_code = exc.code
            return "trial"
        if isinstance(exc, MissingAPIKey):
            return "key"
        return _failure_kind(exc)

    def _refine(self, text: str, context, vocab, language: str = "", **practice) -> RefineResult:
        if self.prefs.trial_used_up and getattr(self.config.refine, "preset", "") == trial.PRESET:
            # The trial server already said this Mac's share is spent: don't ask it again.
            raise TrialUnavailable("trial_used_up", "")
        return self.refiner.refine(text, context, vocab=vocab, language=language, **practice)

    def _note_trial(self, fallback: str) -> None:
        """The trial server said no: keep the menu's "enter your own key" up until one is saved."""

        if fallback == "trial":
            self._trial_over = True
            if getattr(self, "_trial_code", "") == "trial_used_up" and not self.prefs.trial_used_up:
                self.set_preference("trial_used_up", True)

    def _show_raw_key(self, fallback: str) -> None:
        self._note_trial(fallback)
        if fallback == "trial":
            self.capsule.show("inserted-raw-trial", why=getattr(self, "_trial_code", ""))
        else:
            self.capsule.show("inserted-raw-key")

    def _refresh_issues(self) -> tuple[str, ...]:
        menubar = getattr(self, "menubar", None)
        if menubar is None:
            return ()
        issues = self._current_issues()
        menubar.set_issues(issues)
        return issues

    def _menu_snapshot(self) -> Snapshot:
        """Everything the menu shows, read as it opens."""

        config = self.config
        jarvis = getattr(config, "jarvis_config", {}) or {}
        user_paths = getattr(config, "user_paths", None)
        history = user_paths.trace_db_path if user_paths is not None and self.prefs.save_history else None
        if not getattr(getattr(self, "recorder", None), "wedged", False):
            devices.refresh_if_changed()
        recent = getattr(self, "_recent", None)
        return Snapshot(
            state=self.state,
            issues=self._refresh_issues(),
            recent=recent,
            active_preset=config.refine.preset,
            inputs=tuple(devices.list_input_devices()),
            active_input=getattr(config, "input_device", "") or "",
            refine=self.prefs.refine,
            english_practice=self.prefs.english_practice,
            usage=usage.today_line(history, prices(jarvis)) if history is not None else "",
        )

    def _remember(self, text: str, app: str = "") -> None:
        """The menu's "最近一次" and the clipboard fallback."""

        self._copy_fallback_text = text
        self._recent = Recent(text=text, app=app, at=time.time())

    def show_log(self) -> None:
        user_paths = getattr(self.config, "user_paths", None)
        if user_paths is None:
            return
        try:
            user_paths.log_path.parent.mkdir(parents=True, exist_ok=True)
            user_paths.log_path.touch(exist_ok=True)
            subprocess.Popen(["/usr/bin/open", str(user_paths.log_path)])
        except Exception:
            LOGGER.warning("Could not open the log", exc_info=True)

    def export_diagnostics(self) -> None:
        """Zip the log, the changed settings and a summary for a bug report, and show it in Finder."""

        user_paths = getattr(self.config, "user_paths", None)
        if user_paths is None:
            return
        summary = self._diagnostic_summary()
        secrets = [os.environ.get(name, "") for name in api_key_names(self.config)]

        def run() -> None:
            try:
                # Inside ~/.typlus rather than on the Desktop, which macOS guards
                # with a permission prompt of its own.
                path = diagnostics.export(
                    user_paths.config_dir / "diagnostics",
                    name=brand.ENGLISH_NAME,
                    version=app_version(),
                    summary=summary,
                    log_path=user_paths.log_path,
                    user_config_path=user_paths.user_config_path,
                    secrets=secrets,
                )
                LOGGER.info("Diagnostics written to %s", path)
                subprocess.Popen(["/usr/bin/open", "-R", str(path)])
            except Exception:
                LOGGER.exception("Could not export diagnostics")

        threading.Thread(target=run, daemon=True, name="diagnostics").start()

    def _diagnostic_summary(self) -> dict:
        """What a bug report needs to know about this Mac and this install. No keys."""

        import platform  # noqa: PLC0415

        def safe(read):
            try:
                return read()
            except Exception as exc:
                return f"unavailable: {exc!r}"

        config = self.config
        asr_config = (config.jarvis_config or {}).get("asr") or {}
        repo_id = mlx_whisper_repo(asr_config)
        return {
            "app": {"version": app_version(), "state": self.state, "issues": safe(lambda: list(self._current_issues()))},
            "mac": {
                "macos": platform.mac_ver()[0],
                "chip": reach.machine(),
                "mainland_china": reach.in_mainland_china(),
                "interface_language": i18n.current(),
                "keycodes": {"v": keyboard_layout.keycode("v"), "z": keyboard_layout.keycode("z")},
            },
            "permissions": {
                "microphone": safe(permissions.microphone_status),
                "accessibility": safe(has_accessibility_trust),
            },
            "audio": {
                "inputs": safe(devices.list_input_devices),
                "chosen": config.input_device or "(system default)",
                "system_default": safe(devices.current_input_device),
                "input_channel": config.input_channel,
            },
            "speech": {
                "provider": asr_config.get("provider"),
                "model": repo_id,
                "cached": safe(lambda: model_is_cached(repo_id)),
                "download_from": os.environ.get("HF_ENDPOINT", ""),
                "language": asr_config.get("language") or "auto",
            },
            "refine": {
                "preset": config.refine.preset,
                "model": config.refine.model,
                "base_url": config.refine.base_url,
                "keys_set": {name: bool(os.environ.get(name)) for name in api_key_names(config)},
            },
            "preferences": self.prefs.to_dict(),
        }

    def _start_model_prefetch(self) -> None:
        """Pull the ASR weights now, with the capsule showing progress, instead of
        inside the first F5 where a 1.5 GB download looks like the app has hung."""

        asr_config = self.config.jarvis_config.get("asr") or {}
        provider = str(asr_config.get("provider") or "").strip().lower()
        repo_id = mlx_whisper_repo(asr_config)
        if provider != "mlx_whisper" or model_is_cached(repo_id):
            self._warm_up_asr()
            return

        started = time.monotonic()
        self._download = (0.0, started)
        self._model_failed = ""
        self._refresh_issues()

        def report(fraction: float) -> None:
            self._download = (fraction, started)
            self._download_progress(fraction, _eta_text(fraction, time.monotonic() - started))
            with self._lock:
                # Only while nothing else is on screen: the progress used to
                # replace a dictation in the middle of it. The guide shows its own.
                if self.state == "idle" and self.capsule.state in ("hidden", "download") and not self._guide_open():
                    self._show_download()

        def run() -> None:
            try:
                report(0.0)
                download_model(repo_id, report)
                LOGGER.info("ASR model %s is ready", repo_id)
                self._download = None
                self._download_progress(1.0, done=True)
                self._warm_up_asr()
                with self._lock:
                    if self.state == "idle" and self.capsule.state in ("hidden", "download") and not self._guide_open():
                        # It downloaded while they did something else: say it can be used now.
                        self.capsule.show("ready", name=brand.display_name())
            except Exception as exc:
                # The menu offers to try again; the app stays up meanwhile.
                LOGGER.exception("Model prefetch failed for %s", repo_id)
                self._download = None
                self._model_failed = "disk" if isinstance(exc, NotEnoughDisk) else "net"
                self._refresh_issues()
                self._download_progress(0.0, error=True, why=self._model_failed)
            finally:
                self._download = None
                self.capsule.hide_if("download")

        threading.Thread(target=run, daemon=True, name="model-prefetch").start()

    def _guide_open(self) -> bool:
        windows = getattr(self, "windows", None)
        return windows is not None and windows._visible("onboarding")

    def _download_progress(self, fraction: float, eta: str = "", done: bool = False, error: bool = False, why: str = "") -> None:
        """The guide's progress bar, when it is open."""

        windows = getattr(self, "windows", None)
        if windows is not None:
            self._call_ui(windows.download, fraction, eta, done, error, why)

    def _show_download(self) -> None:
        download = getattr(self, "_download", None)
        if download is None:
            return
        fraction, started = download
        self.capsule.show("download", p=round(fraction, 3), eta=_eta_text(fraction, time.monotonic() - started))

    def _prime_microphone(self) -> None:
        """Do the first recording's device setup now, in the background.

        Holds the hotkey lock meanwhile, so an F5 in the first second or two
        after launch waits for it rather than racing a PortAudio re-init.
        """

        if permissions.microphone_status() != "authorized":
            return  # opening a stream would ask for the microphone before onboarding explains why

        def run() -> None:
            started = time.monotonic()
            with self._lock:
                try:
                    self._select_capture_device()
                    devices.prime_input(self.recorder.device, int(self.config.sample_rate))
                except Exception:
                    LOGGER.warning("Priming the microphone failed; the first dictation opens it cold", exc_info=True)
                    return
            LOGGER.info("Microphone primed in %.2fs", time.monotonic() - started)

        threading.Thread(target=run, daemon=True, name="mic-prime").start()

    def _warm_up_asr(self) -> None:
        """Load the recognizer in the background before the first dictation.

        Queued on the processing worker so it can never run alongside a real
        transcription; a dictation made meanwhile waits for the load it would
        have had to do anyway.
        """

        warmup = getattr(self.asr, "warmup", None)
        if warmup is not None:
            self.executor.submit(warmup)

    def _set_menubar(self, state: str) -> None:
        """Update the menu-bar status icon. No-op when menubar is unavailable."""

        if getattr(self, "headless", False):
            return
        mb = getattr(self, "menubar", None)
        if mb is not None:
            mb.set_state(state)

    def start(self) -> None:
        """Start the app."""

        self.overlay.setup()
        install_main_menu()
        self._apply_preferences()
        # On first launch the guide asks for each permission after saying why.
        guided = not self.prefs.onboarding_done and getattr(self, "windows", None) is not None
        if not has_accessibility_trust():
            LOGGER.warning("Accessibility permission is not granted; Fn capture/paste may fail.")
            if not guided:
                request_accessibility_trust()
        menubar = getattr(self, "menubar", None)
        if menubar is not None:
            menubar.setup()
            menubar.set_state("idle")
        if not self._start_hotkeys():
            # Picked up by itself once Accessibility is granted: no restart.
            if not guided:
                self.capsule.show("perm")
            AppHelper.callLater(HOTKEY_RETRY_S, self._retry_hotkeys)
        self._refresh_issues()
        self._prime_microphone()
        self._start_model_prefetch()
        AppHelper.callLater(STATS_FIRST_S, self._send_stats)
        # Deferred onto the run loop: this app is LSUIElement, and before
        # -[NSApplication run] it is not active yet, so a window can open
        # behind whatever the user is looking at or not come up at all.
        AppHelper.callLater(0.3, self._first_run)
        LOGGER.info("%s ready. Tap right Cmd to start/stop dictation.", brand.ENGLISH_NAME)

    def _start_hotkeys(self) -> bool:
        try:
            # Creating the event tap untrusted makes macOS pop its own
            # Accessibility prompt, ahead of the guide that explains it.
            if not has_accessibility_trust():
                raise RuntimeError("Accessibility is not granted")
            self.hotkeys.start()
            return True
        except RuntimeError:
            if not getattr(self, "_hotkeys_failed", False):
                LOGGER.warning("Global hotkey monitor unavailable until Accessibility is granted")
            self._hotkeys_failed = True
            return False

    def _retry_hotkeys(self) -> None:
        if self._start_hotkeys():
            LOGGER.info("Accessibility granted; hotkeys are live")
            self.capsule.hide_if("perm")
            self._refresh_issues()
            return
        AppHelper.callLater(HOTKEY_RETRY_S, self._retry_hotkeys)

    # ----------------------------------------------------------------- keys

    def _on_hotkey(self, action: str) -> None:
        # Typing is reported from inside the keyboard event tap, so these two
        # never wait on the app lock.
        if action == "typed":
            self._on_typed()
            return
        if action == "undo":
            self._on_user_undo()
            return
        guide = {"hold_start": "primary_down", "hold_abort": "cancel"}.get(action, action)
        if guide in ("primary_down", "hands_free", "cancel"):
            self._tell_guide({"t": "hotkey", "a": guide})
        with self._lock:
            if action == "cancel":
                self._cancel()
                return
            if action == "send":
                # Enter while recording: finish like a normal stop, then press
                # Return after the paste. Tied to this dictation's session id.
                if self.state == "recording":
                    self._send_session = getattr(self, "_active_session_id", 0)
                    self._finish_recording()
                return
            if action == "hold_start":
                self._on_hold_start()
                return
            if action == "hold_end":
                if self._hold_is_current():
                    self._finish_recording()
                return
            if action == "hold_abort":
                # Right Cmd became a chord: drop the recording only if this
                # hold started it, never one already running or processing.
                if self._hold_is_current():
                    self._cancel()
                return
            if action == "hands_free":
                if self.state == "idle":
                    self._start_recording("hands_free")
                elif self.state == "recording" and self.mode == "tap":
                    self.mode = "hands_free"
                    self._stop_holding()
                    self._show_recording_ui()
                elif self.state == "recording" and self.mode == "hands_free":
                    self._finish_recording()
                return
            if action == "primary_down":
                self._on_primary_down()
                return
            if action == "primary_up":
                self._on_primary_up()
                return
            if action == "primary":
                if self.state == "idle":
                    self._start_recording("tap")
                elif self.state == "recording":
                    self._finish_recording()

    def _on_primary_down(self) -> None:
        # Use the event-generation time captured by the hotkey monitor, not
        # time.monotonic() here. Handler time is unreliable: _start_recording
        # blocks the runloop briefly while the mic opens, so KeyUp can wait in
        # the queue and look like a long press, misfiring hold-to-talk.
        now = self.hotkeys.last_primary_down_at
        self._primary_down_at = now
        finished = getattr(self, "_finished_by_tap", None)
        if finished is not None and 0.0 <= now - finished[0] <= DOUBLE_CLICK_SECONDS:
            # Tapped again right after the tap that finished a dictation:
            # send it rather than start another.
            self._finished_by_tap = None
            self._send_when_pasted(finished[1])
            return
        gap = now - getattr(self, "_last_short_tap_at", 0.0)
        LOGGER.debug("HOTKEY-DEBUG primary_down now=%.3f last_short_tap=%.3f gap=%.3f state=%s mode=%s", now, getattr(self, "_last_short_tap_at", 0.0), gap, self.state, self.mode)
        # A tap cannot land before the one it follows, so a negative gap means
        # the two timestamps came from different clocks rather than that this
        # was a double tap. Treating it as one wedges every later press into
        # the double-tap branch, where a second press only re-renders and never
        # stops the recording, for the life of the process.
        if 0.0 <= gap <= DOUBLE_CLICK_SECONDS:
            if self.state == "idle":
                self._start_recording("hands_free")
            elif self.state == "recording":
                self.mode = "hands_free"
                self._stop_holding()
                self._show_recording_ui()
            return

        if self.state == "idle":
            self._start_recording("tap")
            if self.state == "recording":
                self._arm_hold_timer(now)
        elif self.state == "recording" and self.mode in ("tap", "hands_free"):
            self._finished_by_tap = (now, getattr(self, "_active_session_id", 0))
            self._finish_recording()

    def _on_hold_start(self) -> None:
        """Right Cmd held alone: record until it is let go. Only ever starts a recording."""

        if self.state != "idle":
            return
        self._start_recording("tap", hold=True)
        if self.state != "recording":
            return
        self._hold_session_id = self._active_session_id
        self._primary_down_at = self.hotkeys.last_primary_down_at
        self._holding = True
        self._show_recording_ui()
        # Ducking and pausing music wait until the hold has outlasted a quick
        # chord (Cmd-click, Cmd+C), which would otherwise blip the music.
        timer = threading.Timer(HOLD_CONFIRM_S, partial(self._on_hold_confirmed, self._active_session_id))
        timer.daemon = True
        timer.start()

    def _on_hold_confirmed(self, session_id: int) -> None:
        with self._lock:
            if self.state == "recording" and session_id == getattr(self, "_active_session_id", 0):
                self._duck_for(getattr(self, "_capture_device", "") or "")
                if getattr(self, "_ducked", False):
                    self._show_recording_ui()

    def _hold_is_current(self) -> bool:
        session = getattr(self, "_active_session_id", 0)
        return self.state in ("starting", "recording") and session == getattr(self, "_hold_session_id", -1)

    def _on_primary_up(self) -> None:
        if self.state != "recording" or self.mode != "tap":
            return

        up_at = self.hotkeys.last_primary_up_at
        held_for = up_at - getattr(self, "_primary_down_at", 0.0)
        LOGGER.debug("HOTKEY-DEBUG primary_up up_at=%.3f down_at=%.3f held_for=%.3f", up_at, getattr(self, "_primary_down_at", 0.0), held_for)
        if held_for < LONG_PRESS_SECONDS:
            self._last_short_tap_at = up_at
            if getattr(self, "_holding", False):
                # The hold view went up before this release was read (the run
                # loop was busy opening the mic): it was a tap after all.
                self._stop_holding()
                self._show_recording_ui()
            return

        self._finish_recording()

    def _arm_hold_timer(self, down_at: float) -> None:
        """Switch the capsule to "release to finish" once the key has been held long enough."""

        self._cancel_hold_timer()
        timer = threading.Timer(LONG_PRESS_SECONDS, partial(self._on_hold_timer, self._active_session_id, down_at))
        timer.daemon = True
        self._hold_timer = timer
        timer.start()

    def _on_hold_timer(self, session_id: int, down_at: float) -> None:
        with self._lock:
            self._hold_timer = None
            hotkeys = getattr(self, "hotkeys", None)
            released = hotkeys is None or hotkeys.last_primary_up_at >= down_at
            if released or self.state != "recording" or self.mode != "tap":
                return
            if session_id != getattr(self, "_active_session_id", 0):
                return
            self._holding = True
            self._show_recording_ui()

    def _cancel_hold_timer(self) -> None:
        timer = getattr(self, "_hold_timer", None)
        if timer is not None:
            timer.cancel()
        self._hold_timer = None

    def _stop_holding(self) -> None:
        self._cancel_hold_timer()
        self._holding = False

    def _on_typed(self) -> None:
        """The user typed after a paste: undo or replace would now take back their typing."""

        self._keys_wanted = False
        self._insertion = None
        self.capsule.hide_if(*INSERTED_STATES)

    def _on_user_undo(self) -> None:
        """The user pressed Cmd+Z themselves right after a paste; the capsule confirms it."""

        insertion = self._insertion
        if insertion is not None and frontmost_pid() != insertion.pid:
            # Pressed in another app: it took back nothing of the paste.
            self._on_typed()
            return
        self._keys_wanted = False
        self._insertion = None
        if self.capsule.state in INSERTED_STATES:
            self.capsule.show("undone")

    # -------------------------------------------------------------- capsule

    def _on_overlay_action(self, action: str, data: dict | None = None) -> None:
        data = data or {}
        text = str(data.get("text") or "")
        if action == "draft":
            self._on_card_draft(text)
            return
        if action == "field":
            # While the card's field has the keyboard, keys pressed are for the
            # card, not typing after the paste.
            self._keys_wanted = self._insertion is not None and not data.get("focus")
            return
        if action == "word":
            self._toggle_word(data)
            return
        with self._lock:
            if action == "primary":
                if self.state == "idle":
                    self._start_recording("tap")
            elif action == "cancel":
                self._cancel()
            elif action == "finish":
                if self.state == "recording":
                    self._finish_recording()
            elif action == "extend":
                self._extend_recording()
            elif action == "undo":
                self._undo_insertion()
            elif action == "edit":
                self._edit_insertion()
            elif action == "rerefine":
                self._rerefine_insertion()
            elif action == "replace":
                self._replace_insertion(text)
            elif action == "useen":
                self._use_english(text)
            elif action == "copyen":
                if text:
                    set_clipboard_text(text)
                self.capsule.show("copied")
            elif action == "done":
                self._commit_edit(text)
            elif action == "close":
                self._end_edit()
            elif action in ("setkey", "billing", "input", "micperm", "perm", "log", "restart"):
                self.capsule.hide()
                if action == "restart":
                    self._restart()
                elif action == "setkey":
                    self.open_settings("model")
                elif action == "billing":
                    permissions.open_url(OPENAI_BILLING_URL)
                elif action == "input":
                    self.open_settings("audio")
                elif action == "micperm":
                    permissions.open_url(permissions.MICROPHONE_SETTINGS)
                elif action == "perm":
                    permissions.open_url(permissions.ACCESSIBILITY_SETTINGS)
                else:
                    self.show_log()

    def _on_overlay_hover(self, on: bool) -> None:
        self.capsule.set_hover(on)

    def _on_card_draft(self, text: str) -> None:
        if self.capsule.state == "edit-notarget":
            # Clipboard only: _copy_fallback_text stays the original so the
            # correction recorded on commit is measured against what the
            # dictation actually produced, not against the last keystroke.
            set_clipboard_text(text)

    def _undo_insertion(self) -> None:
        """The capsule's Undo: send Cmd+Z to the app the text went to, if it is still in front."""

        insertion, self._insertion = self._insertion, None
        self._keys_wanted = False
        if insertion is None:
            self.capsule.hide()
            return
        if frontmost_pid() != insertion.pid:
            self.capsule.show("notice", msg=t("目标 App 已切换，没法撤销", "You switched apps, so it can't be undone"))
            return
        undo_last_edit()
        self.capsule.show("undone")

    def _edit_insertion(self) -> None:
        insertion = self._insertion
        if insertion is None:
            self.capsule.hide()
            return
        self._keys_wanted = False
        self.capsule.show("edit-modify", text=insertion.text)

    def _replace_insertion(self, text: str) -> None:
        """The card's Replace: take the pasted text back and paste the edited text."""

        corrected = text.strip()
        insertion = self._insertion
        self.capsule.end_edit()
        if not corrected or (insertion is not None and corrected == insertion.text):
            self.capsule.hide()
            return
        if insertion is None:
            # Something was typed after the paste; undoing now would take that back.
            set_clipboard_text(corrected)
            self.capsule.show(
                "notice", msg=t("原文已经改动过，修改后的文字已复制", "The text changed since; your edit is copied")
            )
            return
        self._keys_wanted = False
        threading.Thread(
            target=self._swap_text, args=(insertion, corrected, True), daemon=True, name="replace"
        ).start()

    def _toggle_word(self, data: dict) -> None:
        """A word or phrase clicked on the English card: save it to the word book, or take it back."""

        card = getattr(self, "_english_card", None)
        trace = getattr(self, "trace", None)
        term = str(data.get("term") or "").strip()
        if not card or trace is None or not term:
            return
        db = trace.db_path
        if not data.get("on"):
            word_book.remove_word(db, term, card["en"])
            self._windows_changed(history=True)
            return
        phrase = data.get("kind") == "phrase"
        meaning = card["zh"].get(term)
        session_id = getattr(self, "_last_trace_id", None) if self.prefs.save_history else None
        word_id = word_book.add_word(db, term, "phrase" if phrase else "word", meaning, card["en"], card["source"], session_id)
        self._windows_changed(history=True)
        if word_id and not meaning:
            threading.Thread(target=self._lookup_meaning, args=(db, word_id, term, card["en"], card["source"]), daemon=True, name="word-meaning").start()

    def _lookup_meaning(self, db, word_id: int, term: str, sentence: str, source: str = "") -> None:
        try:
            meaning = self.refiner.define(term, sentence, source)
            if meaning and word_book.set_word_meaning(db, word_id, meaning):
                self._windows_changed(history=True)
        except Exception:
            LOGGER.warning("Word meaning lookup failed for %r", term, exc_info=True)

    def _use_english(self, text: str) -> None:
        """The English card's "Use this": the same swap as Replace, with the English version."""

        insertion = self._insertion
        if insertion is None or not text:
            self.capsule.hide()
            return
        self._keys_wanted = False
        threading.Thread(target=self._swap_text, args=(insertion, text, False), daemon=True, name="use-english").start()

    def _rerefine_insertion(self) -> None:
        """The capsule's Re-refine: try refinement again on what was pasted raw."""

        insertion = self._insertion
        if insertion is None or self.state != "idle":
            self.capsule.hide()
            return
        self.capsule.show("refining", raw=insertion.raw)
        future = self.executor.submit(self._rerefine, insertion, getattr(self, "_active_session_id", 0))
        future.add_done_callback(self._log_processing_done)

    def _rerefine(self, insertion: Insertion, session_id: int) -> None:
        fallback = ""
        try:
            result = self._refine(
                insertion.raw, self._refine_context(insertion.context), vocab=getattr(self, "vocab", []) or []
            )
            fallback = getattr(result, "fallback", "") or ""
        except Exception as exc:
            LOGGER.warning("Refinement failed again", exc_info=True)
            result = None
            fallback = self._fallback_for(exc)
        with self._lock:
            if session_id != getattr(self, "_active_session_id", 0) or self.state != "idle":
                return  # a new dictation has the capsule now
            current = self._insertion is insertion
        if result is None or fallback:
            if not current:
                self.capsule.hide()
            elif fallback in ("key", "trial"):
                self._show_raw_key(fallback)
            else:
                self.capsule.show("inserted-raw-net", why=fallback)
            return
        self._swap_text(insertion, result.text, False)

    def _swap_text(self, insertion: Insertion, new_text: str, from_card: bool) -> None:
        """Take ``insertion`` back in its app and paste ``new_text`` in its place.

        Runs off the main thread, and never holds the app lock while it waits:
        the keyboard event tap takes that lock.
        """

        if from_card:
            time.sleep(KEY_HANDBACK_S)
        with self._lock:
            current = self._insertion is insertion and self.state == "idle"
        if not current or frontmost_pid() != insertion.pid:
            set_clipboard_text(new_text)
            message = (
                t("目标 App 已切换，新文字已复制", "You switched apps; the new text is copied")
                if current
                else t("原文已经改动过，新文字已复制", "The text changed since; the new text is copied")
            )
            self.capsule.show("notice", msg=message)
            return
        undo_last_edit()
        time.sleep(UNDO_SETTLE_S)
        paste_text(new_text)
        with self._lock:
            self._insertion = dataclasses.replace(insertion, text=new_text)
            self._keys_wanted = True
            self._remember(new_text, insertion.context.app_name)
        self.capsule.show("replaced", n=count_units(new_text))
        if from_card:
            self._record_correction(insertion.text, new_text)

    def _commit_edit(self, text: str) -> None:
        """The card's Done, where there was nowhere to paste: the clipboard gets the edit."""

        before = getattr(self, "_copy_fallback_text", "")
        corrected = text.strip()
        self.capsule.end_edit()
        if not corrected:
            self.capsule.hide()
            return
        set_clipboard_text(corrected)
        if corrected != before:
            self._record_correction(before, corrected)
        recent = getattr(self, "_recent", None)
        self._remember(corrected, recent.app if recent else "")
        self.capsule.show("copied")

    def _end_edit(self) -> None:
        """Close the card, leaving the clipboard as the card left it."""

        self.capsule.end_edit()
        self.capsule.hide()

    def _record_correction(self, before: str, after: str) -> None:
        user_paths = getattr(self.config, "user_paths", None)
        if user_paths is not None and self.prefs.save_history:
            append_correction(user_paths.corrections_path, getattr(self, "_last_trace_id", None), before, after)
        LOGGER.info("Correction recorded: %d chars -> %d chars", len(before), len(after))

    # ------------------------------------------------------------ recording

    def _store_sent_text(self, session_id: int, pasted: str, text: str) -> None:
        """A pasted dictation went out as ``text`` (from the sent-text watcher's thread)."""

        try:
            user_paths = getattr(self.config, "user_paths", None)
            if user_paths is not None and self.prefs.save_history and self.prefs.save_sent_text:
                if set_sent_text(user_paths.trace_db_path, session_id, text):
                    self._windows_changed(history=True)
            self._learn_from_edit(pasted, text)
        except Exception:
            LOGGER.warning("Could not handle the sent text", exc_info=True)

    def _learn_from_edit(self, pasted: str, sent: str) -> None:
        """Words fixed by hand before sending go into the vocabulary."""

        user_paths = getattr(self.config, "user_paths", None)
        if user_paths is None:
            return
        learned = sum(learn_term(user_paths.vocab_path, right, wrong) for wrong, right in learned_fixes(pasted, sent))
        if learned:
            self.reload_vocab()
            LOGGER.info("Learned %d word(s) from a hand edit", learned)
            self._windows_changed()

    def _start_recording(self, mode: Mode, hold: bool = False) -> None:
        if getattr(self, "_download", None) is not None:
            # Nothing to transcribe with until the model is here; say so
            # instead of recording into a wait of several minutes.
            self._show_download()
            return
        if permissions.microphone_status() in ("denied", "restricted"):
            self._refresh_issues()
            self.capsule.show("mic", why="denied")
            return
        self._active_session_id = getattr(self, "_active_session_id", 0) + 1
        self._insertion = None
        self._keys_wanted = False
        self._send_session = None
        watcher = getattr(self, "sent_watcher", None)
        if watcher is not None:
            watcher.cancel()
        self.mode = mode
        self._holding = False
        self.state = "starting"
        self._set_menubar("starting")
        self._recording_started_at = time.monotonic()
        self._recording_limit_s = float(getattr(self.config, "max_recording_seconds", 0.0) or 0.0)
        caret = self.prefs.capsule_position == "caret"
        # The microphone opens first: everything that used to come before it
        # (focus probing, re-enumerating devices, two osascript runs to mute)
        # delayed it by enough to lose the first words, and ran inside the
        # keyboard event tap, stalling typing system-wide meanwhile.
        capture = self._select_capture_device()
        if not caret:
            # At the caret the capsule waits for the caret's position, which
            # is only read once the microphone is open.
            self.capsule.show("starting")
        self._stretches = []
        self._cloud = self._open_cloud_stream()
        try:
            capture = self._start_microphone(capture)
        except Exception:
            LOGGER.exception("Failed to start microphone")
            self._close_cloud_stream()
            self._restore_audio_ducking()
            self.state = "idle"
            self._set_menubar("idle")
            self._refresh_issues()
            denied = permissions.microphone_status() in ("denied", "restricted")
            stuck = getattr(self.recorder, "wedged", False)
            self.capsule.show("mic", why="denied" if denied else "stuck" if stuck else "busy")
            return
        self._capture_device = capture
        self.focus_context = capture_focus_context(read_before_text=self.prefs.send_before_text)
        if self._cloud is not None:
            self._cloud.give_context(self._qwen_context(self.focus_context))
        if caret:
            self.capsule.set_anchor("caret", caret_rect())
        self._ducked = False
        if not hold:
            self._duck_for(capture)
        self.state = "recording"
        self._set_menubar("recording")
        self._show_recording_ui()
        self._play_sound("Tink")
        self._start_recording_timeout()
        self._prewarm_refiner()
        # Whisper idle for 20 s or more takes about 1.1 s on its next pass
        # instead of 0.4 s, which a short dictation paid in full after the
        # stop. A silent pass queued now on the worker is done while he talks.
        warmup = getattr(getattr(self, "asr", None), "warmup", None)
        executor = getattr(self, "executor", None)
        if warmup is not None and executor is not None and self._cloud is None:
            executor.submit(warmup)

    def _open_cloud_stream(self) -> cloud_asr.QwenStream | None:
        """With Qwen chosen and its key set: a stream the microphone feeds from its first block."""

        stream = None
        engine = self.prefs.asr_engine
        if engine in ("qwen", "qwen_message"):
            stream = cloud_asr.stream_if_set_up(
                getattr(self, "vocab", []) or [],
                int(getattr(self.config, "sample_rate", 16000)),
                model=cloud_asr.MESSAGE_MODEL if engine == "qwen_message" else cloud_asr.MODEL,
                weights=getattr(self, "hotword_weights", None),
                languages=self._qwen_languages(),
                context_pending=True,
            )
        self.recorder.on_chunk = stream.feed if stream is not None else None
        return stream

    def _qwen_languages(self) -> list[str]:
        """The spoken language when one is chosen; on Automatic all of his. None
        until the Mac's own are read: English alone would have Qwen hear Chinese as English."""

        spoken = str((getattr(getattr(self, "asr", None), "_asr_config", None) or {}).get("language") or "")
        if spoken:
            return [spoken]
        return sorted(self._allowed_languages()) if languages.known() else []

    def _qwen_context(self, context: FocusContext) -> list[str]:
        """What Qwen reads before it hears him, oldest first: what it heard in this
        app lately, then the window title and the text before the caret as the
        privacy settings allow. It leans toward words it has seen there."""

        try:
            prefs = self.prefs
            before = str(context.before_text or "") if prefs.send_before_text else ""
            now = time.time()
            texts = [
                heard.text
                for heard in tuple(getattr(self, "_qwen_heard", ()))  # a copy: the worker may be adding one
                if heard.app == context.app_name and now - heard.at <= QWEN_CONTEXT_S and heard.text not in before
            ][-QWEN_CONTEXT_DICTATIONS:]
            if prefs.send_window_title and context.window_title:
                texts.append(str(context.window_title))
            if before:
                texts.append(before)
            return texts
        except Exception:
            LOGGER.debug("Could not gather Qwen's context", exc_info=True)
            return []

    def _close_cloud_stream(self) -> cloud_asr.QwenStream | None:
        """Unhook this dictation's stream from the microphone and hand it over."""

        stream, self._cloud = getattr(self, "_cloud", None), None
        self.recorder.on_chunk = None
        return stream

    def _duck_for(self, capture: str) -> None:
        """Mute the speakers and pause music for this recording, unless a headset makes it pointless."""

        duck = self._speakers_need_ducking(capture)
        if duck:
            self._run_audio_io(self.audio_ducker.duck)
            self._run_audio_io(self._pause_media)
        self._ducked = duck and bool(getattr(self.audio_ducker, "enabled", True))

    def _start_microphone(self, capture: str) -> str:
        """Open the mic; if it fails, re-read the devices once and try again.

        Devices are only re-enumerated when CoreAudio reports a change, so a
        change that slipped past that check surfaces here as a failed open.
        Returns the name of the device actually being captured.
        """

        try:
            self.recorder.start()
            return capture
        except Exception:
            if getattr(self.recorder, "wedged", False):
                raise  # retrying would wait on CoreAudio all over again
            LOGGER.warning("Microphone failed to open; re-reading devices and retrying", exc_info=True)
        devices.refresh()
        capture = self._resolve_capture_device()
        self.recorder.start()
        return capture

    def _run_audio_io(self, job) -> None:
        """Run a system-audio side effect off the hotkey path, in order."""

        executor = getattr(self, "_audio_io", None)
        if executor is None:
            job()
            return
        executor.submit(job)

    def _show_recording_ui(self) -> None:
        if self.mode == "hands_free":
            mode = "latch"
        else:
            mode = "hold" if getattr(self, "_holding", False) else "click"
        focus = getattr(self, "focus_context", None)
        started = getattr(self, "_recording_started_at", 0.0)
        self.capsule.show(
            "rec",
            mode=mode,
            sel=bool(focus and focus.selected_text) and self.prefs.rewrite_selection,
            ducked=bool(getattr(self, "_ducked", False)) and self.prefs.show_ducked,
            max=self._recording_limit(),
            ext=int(EXTEND_RECORDING_S // 60),
            elapsed=round(max(0.0, time.monotonic() - started), 2) if started else 0.0,
        )

    def _recording_limit(self) -> float:
        """This recording's length limit: the setting, plus any 延长 presses."""

        limit = getattr(self, "_recording_limit_s", None)
        if limit is None:
            limit = float(getattr(self.config, "max_recording_seconds", 0.0) or 0.0)
        return float(limit)

    def _play_sound(self, name: str) -> None:
        """The optional start/stop sounds (off by default)."""

        if self.prefs.sounds != "start_end" or getattr(self, "headless", False):
            return

        def play() -> None:
            try:
                from AppKit import NSSound

                sound = NSSound.soundNamed_(name)
                if sound is not None:
                    sound.play()
            except Exception:
                LOGGER.debug("Could not play %s", name, exc_info=True)

        self._call_ui(play)

    def _finish_recording(self) -> None:
        if self.state != "recording":
            return
        if self._defer_finish_until_microphone_ready():
            return
        self._cancel_recording_timeout()
        self._stop_holding()
        session_id = getattr(self, "_active_session_id", 0)
        self.state = "processing"
        self._set_menubar("processing")
        try:
            audio = self.recorder.stop()
        except Exception:
            LOGGER.exception("Failed to stop microphone")
            cloud = self._close_cloud_stream()
            if cloud is not None:
                cloud.cancel()
            self._restore_audio_ducking()
            self.state = "idle"
            self._set_menubar("idle")
            self.capsule.show("error", msg=t("麦克风出错", "Microphone error"))
            return
        self._restore_audio_ducking()
        self._play_sound("Pop")
        self._processing_started_at = time.monotonic()
        self.capsule.show("transcribing")
        stretches, self._stretches = getattr(self, "_stretches", []), []
        heard_until = int(getattr(self.recorder, "heard_until", 0))
        future = self.executor.submit(
            self._process_audio, audio, self.focus_context, session_id, stretches, heard_until, self._close_cloud_stream()
        )
        future.add_done_callback(self._log_processing_done)

    def _cancel(self) -> None:
        was = getattr(self, "state", "idle")
        self._active_session_id = getattr(self, "_active_session_id", 0) + 1
        self._cancel_recording_timeout()
        self._stop_holding()
        if was == "recording":
            try:
                self.recorder.stop()
            except Exception:
                LOGGER.exception("Failed to stop microphone during cancel")
        for stretch in getattr(self, "_stretches", []):
            stretch.cancel()
        self._stretches = []
        cloud = self._close_cloud_stream()
        if cloud is not None:
            cloud.cancel()
        self._restore_audio_ducking()
        self.state = "idle"
        self._set_menubar("idle")
        if was in ("starting", "recording", "processing"):
            self.capsule.show("cancelled")
        else:
            self.capsule.hide()

    def _restore_audio_ducking(self) -> None:
        restore_all = getattr(self.audio_ducker, "restore_all", None)
        self._run_audio_io(restore_all if restore_all is not None else self.audio_ducker.restore)
        self._run_audio_io(self._resume_media)

    def _pause_media(self) -> None:
        """Pause music like Typeless does: a lowered song still reaches a mic without echo cancellation.

        The play/pause key goes to the Now Playing app, and nothing says
        whether that app is playing: a class on Zoom holds output open while
        the music sits paused, and some players (NetEase, Spotify) keep output
        open while paused, so the key could start the music. So the key is
        pressed and the output watched: only an app that stops sending audio
        counts as paused, and is resumed afterwards. One that starts, or no
        visible change at all, gets the key again at once and is left alone.
        """

        self._media_paused = False
        if not getattr(self.audio_ducker, "enabled", True):
            return  # "mute other sound while recording" is off: leave the music alone too
        before = devices.playing_apps()
        if not before or all(devices.is_call_app(pid) for pid in before):
            # Only a call is holding output (a class on Zoom): the key would
            # go to a paused music app, or launch Music, and start it.
            return
        bundles = {pid: devices.bundle_id(pid) for pid in before if not devices.is_call_app(pid)}
        unclear = set(getattr(getattr(self, "prefs", None), "unclear_media_apps", ()))
        if all(bundle in unclear for bundle in bundles.values()):
            # Paused and playing look alike for it, so the key could start it.
            LOGGER.info("Not pausing %s: its play/pause can't be confirmed", sorted(bundles.values()))
            return
        press_play_pause()
        deadline = time.monotonic() + MEDIA_CHECK_S
        while time.monotonic() < deadline:
            now = devices.playing_apps()
            if now - before:
                LOGGER.info("Play/pause started paused media (pid %s); pausing it again", sorted(now - before))
                press_play_pause()
                return
            if before - now:
                LOGGER.info("Pausing media while recording")
                self._media_paused = True
                return
            time.sleep(MEDIA_POLL_S)
        # ponytail: an ambiguous press is undone (a brief blip at worst); per-process levels would tell for sure.
        LOGGER.info("Could not tell what play/pause did; pressing it again")
        press_play_pause()
        self._remember_unclear_media(set(bundles.values()))

    def _remember_unclear_media(self, bundles: set[str]) -> None:
        known = list(getattr(getattr(self, "prefs", None), "unclear_media_apps", ()))
        merged = (known + sorted(b for b in bundles if b and b not in known))[-MAX_UNCLEAR_MEDIA_APPS:]
        if merged == known:
            return
        try:
            self.set_preference("unclear_media_apps", merged)
        except Exception:
            LOGGER.debug("Could not remember unclear media apps", exc_info=True)

    def _resume_media(self) -> None:
        if getattr(self, "_media_paused", False):
            self._media_paused = False
            press_play_pause()

    def _start_recording_timeout(self) -> None:
        """Finish by itself at the length limit; the capsule counts down the last minute."""

        self._cancel_recording_timeout()
        max_seconds = self._recording_limit()
        if max_seconds <= 0:
            return
        self._arm_recording_timer(max_seconds)

    def _arm_recording_timer(self, delay: float) -> None:
        self._recording_timer = threading.Timer(delay, self._finish_recording_after_timeout)
        self._recording_timer.daemon = True
        self._recording_timer.start()

    def _extend_recording(self) -> None:
        """The countdown's 延长 button: fifteen more minutes on this recording."""

        limit = self._recording_limit()
        if self.state != "recording" or limit <= 0:
            return
        self._recording_limit_s = limit + EXTEND_RECORDING_S
        timer = getattr(self, "_recording_timer", None)
        if timer is not None:
            timer.cancel()
        elapsed = time.monotonic() - getattr(self, "_recording_started_at", time.monotonic())
        self._arm_recording_timer(max(1.0, self._recording_limit_s - elapsed))
        self._show_recording_ui()
        LOGGER.info("Recording limit extended to %.0f minutes", self._recording_limit_s / 60)

    def _cancel_recording_timeout(self) -> None:
        recording_timer = getattr(self, "_recording_timer", None)
        if recording_timer is not None:
            recording_timer.cancel()
            self._recording_timer = None
        finish_timer = getattr(self, "_finish_debounce_timer", None)
        if finish_timer is not None:
            finish_timer.cancel()
            self._finish_debounce_timer = None

    def _defer_finish_until_microphone_ready(self) -> bool:
        if getattr(self.recorder, "chunk_count", 1) > 0:
            return False
        elapsed = float(getattr(self.recorder, "elapsed", 0.0) or 0.0)
        min_seconds = max(float(getattr(self.config, "min_recording_seconds", 0.0) or 0.0), MIN_MIC_STARTUP_SECONDS)
        if elapsed <= 0.0 or elapsed >= min_seconds:
            return False
        if getattr(self, "_finish_debounce_timer", None) is not None:
            return True
        delay = max(0.05, min_seconds - elapsed)
        LOGGER.info("Deferring finish %.2fs until microphone input delivers audio", delay)
        timer = threading.Timer(delay, self._finish_recording_after_startup_delay)
        timer.daemon = True
        self._finish_debounce_timer = timer
        timer.start()
        return True

    def _finish_recording_after_startup_delay(self) -> None:
        with self._lock:
            self._finish_debounce_timer = None
            if self.state == "recording":
                self._finish_recording()

    def _finish_recording_after_timeout(self) -> None:
        with self._lock:
            if self.state == "recording":
                LOGGER.info("Maximum recording duration reached; finishing dictation.")
                self._finish_recording()

    # ----------------------------------------------------------- processing

    def _on_stretch(self, stretch: np.ndarray) -> None:
        """Audio thread: a stretch cut at his pause is heard now, ahead of the stop.

        Not while Qwen hears the dictation: if it fails, Whisper hears the whole recording."""

        if getattr(self, "_cloud", None) is not None:
            return
        self._stretches.append(self.executor.submit(self._hear, stretch))
        self._prewarm_refiner()

    def _hear(self, audio: np.ndarray) -> Transcript:
        """One stretch through Whisper; one too quiet for the recording's own gate is not sent."""

        floor = float(getattr(self.config, "low_volume_threshold", 0.003))
        if peak_level(audio, int(getattr(self.config, "sample_rate", 16000))) < floor:
            return Transcript(text="", language="unknown", confidence=0.0)
        return self._whisper(audio)

    def _whisper(self, audio: np.ndarray) -> Transcript:
        """The user's word list goes into Whisper's prompt only for a chunk that
        fits one 30 s window: across windows it used to loop on longer speech.

        With Chinese chosen as the spoken language the prompt also starts with a
        line of Simplified Chinese, so the raw text comes out Simplified even
        when refinement is off or fails. Automatic and English get no hint."""

        sample_rate = int(getattr(self.config, "sample_rate", 16000))
        spoken = str((getattr(self.asr, "_asr_config", None) or {}).get("language") or "")
        if audio.size > WHISPER_WINDOW_S * sample_rate:
            transcript = self.asr.transcribe(audio, initial_prompt=None)
            self._learn_language(transcript, audio.size / sample_rate, spoken)
            return transcript
        prompt = getattr(self, "whisper_prompt", "") or ""
        hint = f"{SIMPLIFIED_HINT} {prompt}".strip()
        transcript = self.asr.transcribe(audio, initial_prompt=(hint if spoken == "zh" else prompt) or None)
        if not spoken and audio.size < languages.SHORT_CLIP_S * sample_rate:
            redo = self._misheard_language(transcript, audio)
            if redo:
                transcript = self.asr.transcribe(
                    audio, initial_prompt=(hint if redo == "zh" else prompt) or None, language=redo
                )
        self._learn_language(transcript, audio.size / sample_rate, spoken)
        return transcript

    def _allowed_languages(self) -> set[str]:
        prefs = getattr(self, "prefs", None)
        return languages.allowed(getattr(prefs, "heard_languages", ()))

    def _misheard_language(self, transcript: Transcript, audio: np.ndarray) -> str | None:
        """Automatic, short clip: Whisper's free guess is usually right, so it costs nothing.
        Only a language outside the user's own is checked again: the likeliest
        allowed one, or None to keep what was heard."""

        heard = str(getattr(transcript, "language", "") or "").lower()
        allowed = self._allowed_languages()
        if heard in allowed or not languages.whisper_codes([heard]):
            return None
        try:
            chosen = languages.pick(self.asr.detect_language(audio), allowed)
        except Exception:
            LOGGER.debug("Language detection for a short clip failed; keeping what Whisper heard", exc_info=True)
            return None
        if chosen:
            LOGGER.info("Whisper heard %s in a short clip; hearing it again as %s", heard, chosen)
        return chosen

    def _learn_language(self, transcript: Transcript, seconds: float, spoken: str) -> None:
        """A confident long dictation in Automatic shows a language the user speaks."""

        language = str(getattr(transcript, "language", "") or "").lower()
        if (
            spoken
            or seconds < languages.SHORT_CLIP_S
            or float(getattr(transcript, "confidence", 0.0) or 0.0) < languages.LEARN_MIN_CONFIDENCE
            or len(str(getattr(transcript, "text", "") or "").strip()) < languages.LEARN_MIN_CHARS
        ):
            return
        heard = list(getattr(getattr(self, "prefs", None), "heard_languages", ()))
        if language in heard or not languages.whisper_codes([language]):
            return
        try:
            self.set_preference("heard_languages", (heard + [language])[-languages.MAX_HEARD_LANGUAGES:])
        except Exception:
            LOGGER.debug("Could not remember language %s", language, exc_info=True)

    def _refine_context(self, context: FocusContext) -> FocusContext:
        """What refinement may see of the focused app, as the privacy settings allow."""

        prefs = self.prefs
        changes: dict = {}
        if not prefs.rewrite_selection and context.selected_text:
            changes["selected_text"] = ""
        if not prefs.send_window_title and (context.app_name or context.window_title):
            changes.update(app_name="", window_title="")
        if not prefs.send_before_text and context.before_text:
            changes["before_text"] = ""
        return dataclasses.replace(context, **changes) if changes else context

    def _process_audio(
        self,
        audio: np.ndarray,
        context: FocusContext,
        session_id: int | None = None,
        stretches: list[Future[Transcript]] | tuple[()] = (),
        heard_until: int = 0,
        cloud: cloud_asr.QwenStream | None = None,
    ) -> None:
        session_id = getattr(self, "_active_session_id", 0) if session_id is None else session_id
        headless = bool(getattr(self, "headless", False))
        prefs = self.prefs
        started = time.time()
        step = "asr"

        sample_rate = int(getattr(self.config, "sample_rate", 16000))
        try:
            audio_rms = float(self.recorder.get_volume_level(audio))
        except Exception:
            audio_rms = 0.0
        refine_model = ""
        refine_cfg = getattr(self.config, "refine", None)
        if refine_cfg is not None and prefs.refine:
            refine_model = str(getattr(refine_cfg, "model", "") or "")

        record = SessionRecord(
            started_at=started,
            audio_duration_s=(audio.size / sample_rate) if sample_rate else 0.0,
            audio_rms=audio_rms,
            audio_sample_rate=sample_rate,
            focus_app=context.app_name or "",
            focus_window=context.window_title or "",
            vocab_terms_used=", ".join(getattr(self, "vocab", []) or []),
            hotwords_count=len(getattr(self, "vocab", []) or []),
            asr_model=str(getattr(self.asr, "model_name", "") or ""),
            refine_model=refine_model,
            app_version=app_version(),
            # Only what refinement was allowed to see, for comparing with and without it.
            before_text=getattr(context, "before_text", "") if prefs.send_before_text else "",
        )

        try:
            LOGGER.info("Processing %.2fs audio", audio.size / sample_rate if sample_rate else 0.0)
            quality_ok, quality_message = self.recorder.is_quality_ok(
                audio,
                min_duration=float(getattr(self.config, "min_recording_seconds", 0.25)),
                low_volume_threshold=float(getattr(self.config, "low_volume_threshold", 0.003)),
            )
            if not quality_ok:
                LOGGER.info("Dropping low-quality audio before ASR: %s", quality_message)
                record.error = f"dropped: {quality_message}"
                if not headless:
                    self._show_empty_then_idle(session_id)
                return

            if not headless and not self._is_current_processing_session(session_id):
                return
            if not headless and prefs.refine:
                self._prewarm_refiner()
            LOGGER.info("Starting ASR")

            asr_start = time.monotonic()
            vocab_terms = getattr(self, "vocab", []) or []
            # Stretches cut at his pauses were heard while he talked (on this
            # executor, so they are done); only the rest is left. Without a cut the whole recording already passed the quality gate;
            # after one, the rest may be only his closing pause.
            transcript = cloud.finish(audio) if cloud is not None else None
            if transcript is not None:
                record.asr_model = cloud.model
                heard = getattr(self, "_qwen_heard", None)
                if heard is not None and transcript.text:
                    heard.append(Recent(text=transcript.text, app=context.app_name, at=time.time()))
            elif cloud is not None:
                transcript = self._whisper(audio)  # no stretches were heard while Qwen was listening
            else:
                parts = [stretch.result() for stretch in stretches]
                rest = audio[heard_until:]
                parts.append(self._hear(rest) if parts else self._whisper(rest))
                transcript = _join(parts)
            record.raw_asr_text = transcript.text
            record.raw_asr_language = transcript.language
            record.raw_asr_confidence = float(getattr(transcript, "confidence", 0.0) or 0.0)
            record.latency_asr_ms = int((time.monotonic() - asr_start) * 1000)
            LOGGER.info(
                "ASR result language=%s confidence=%.2f text=%r",
                transcript.language,
                transcript.confidence,
                transcript.text,
            )
            if self._should_drop_transcript(transcript):
                record.error = "dropped: empty/short transcript"
                if not headless:
                    self._show_empty_then_idle(session_id)
                return

            if not headless and not self._is_current_processing_session(session_id):
                return
            fallback = ""
            delivered: list[str] = []
            if prefs.refine:
                step = "refine"
                if not headless:
                    self.capsule.show("refining", raw=transcript.text)
                LOGGER.info("Starting refinement")
                refine_start = time.monotonic()
                practice = {}
                if prefs.english_practice and not headless:
                    # The refined text is pasted the moment the model has written it,
                    # while the English version is still arriving.
                    def paste_early(text: str) -> None:
                        if self._is_current_processing_session(session_id):
                            delivered.append(text)
                            self._deliver(text, transcript.text, context, "", record)

                    practice = {"english": True, "on_text": paste_early}
                try:
                    refined = self._refine(
                        transcript.text,
                        self._refine_context(context),
                        vocab=vocab_terms,
                        language=str(getattr(transcript, "language", "") or ""),
                        **practice,
                    )
                except Exception as exc:
                    # The transcript is already in hand; losing the whole dictation
                    # because the polish step failed is the worst outcome available.
                    LOGGER.exception("Refinement failed; pasting the raw transcript")
                    record.error = f"refine failed, pasted raw transcript: {exc!r}"
                    fallback = self._fallback_for(exc)
                    refined = RefineResult(
                        text=transcript.text, raw_text=transcript.text, model=refine_model, fallback=fallback
                    )
                else:
                    fallback = getattr(refined, "fallback", "") or ""
                    record.prompt_tokens = int(getattr(refined, "prompt_tokens", 0) or 0)
                    record.cached_tokens = int(getattr(refined, "cached_tokens", 0) or 0)
                    record.completion_tokens = int(getattr(refined, "completion_tokens", 0) or 0)
                    if fallback:
                        record.error = f"refine {fallback}, pasted raw transcript"
                record.latency_refine_ms = int((time.monotonic() - refine_start) * 1000)
            else:
                refined = RefineResult(text=transcript.text, raw_text=transcript.text, model="")
            record.refined_text = refined.text or transcript.text
            final_text = record.refined_text

            if headless:
                return  # tests stop here; no overlay/paste path

            if delivered:
                record.refined_text = final_text = delivered[0]
                self._show_english(refined.english, record, session_id)
                return
            if not self._is_current_processing_session(session_id):
                return
            step = "paste"
            self._deliver(final_text, transcript.text, context, fallback, record)
        except Exception as exc:
            record.error = repr(exc)
            LOGGER.exception("Dictation failed")
            if not headless:
                if self._is_current_processing_session(session_id):
                    self.state = "idle"
                    self._set_menubar("idle")
                    self.capsule.show("error", msg=t(*_FAILED_STEP.get(step, ("处理失败", "Something failed"))))
                return
            raise
        finally:
            if cloud is not None:
                cloud.cancel()  # a dictation dropped before recognition leaves its stream open
            record.ended_at = time.time()
            record.latency_total_ms = int((record.ended_at - started) * 1000)
            self._count_dictation(record)
            if record.raw_asr_text:
                # The guide's practice page shows what was heard next to what went in.
                self._tell_guide({"t": "result", "raw": record.raw_asr_text, "text": record.refined_text or record.raw_asr_text})
            trace = getattr(self, "trace", None)
            if trace is not None and prefs.save_history:
                self._last_trace_id = trace.log(record)
                self._windows_changed(history=True)
                watcher = getattr(self, "sent_watcher", None)
                insertion = getattr(self, "_insertion", None)
                if watcher is not None and record.was_pasted and prefs.save_sent_text and insertion is not None:
                    watcher.watch(self._last_trace_id, record.refined_text, insertion.pid)
            self._keep_recording(audio, sample_rate, started)

    def _show_english(self, english: dict | None, record: SessionRecord, session_id: int) -> None:
        """After the paste: keep the English version in the trace and put it in the card."""

        if not english:
            return
        record.english = json.dumps(english, ensure_ascii=False)
        # Not once a new dictation has begun. Typing after the paste is fine:
        # only the swap needs the paste still intact.
        if self.state != "idle" or session_id != getattr(self, "_active_session_id", 0):
            LOGGER.info("English card skipped: a new dictation has begun")
            return
        LOGGER.info("Showing the English card")
        self._english_card = {
            "en": english["en"], "source": record.refined_text or "",
            "zh": {p["en"]: p["zh"] for p in english["phrases"]},
        }
        swap = self._insertion is not None and record.raw_asr_language == "en"
        self.capsule.show(
            "english", use=swap and english["en"].strip() != (record.refined_text or "").strip(), en=english["en"], phrases=english["phrases"]
        )

    def _deliver(self, text: str, raw: str, context: FocusContext, fallback: str, record: SessionRecord) -> None:
        """Paste the text where the caret is, or hand it over in a card when there is nowhere to paste."""

        where = prepare_paste(context) if context.can_insert_text else "none"
        if where in ("ok", "blind"):
            LOGGER.info("Pasting refined text into focused app: %s", context.app_name or "unknown")
            # Measured only where it can mean something: replacing a selection
            # may leave the length as it was.
            before = focused_text_length(context.pid) if where == "ok" and not context.selected_text else None
            # The text stays on the clipboard, so a Cmd+V the target app
            # swallowed is recovered with one manual paste.
            paste_text(text)
            if before is not None and not self._paste_landed(context.pid, before):
                # Claude and other Electron apps can drop focus from their
                # composer while the dictation is processed: never say
                # "inserted" for text that went nowhere.
                LOGGER.warning("The paste changed nothing in %s; the text is on the clipboard", context.app_name or "unknown")
                record.was_pasted = False
                self.state = "idle"
                self._set_menubar("idle")
                self._remember(text, context.app_name)
                self.capsule.show("inserted-unsure")
                self._maybe_send(fallback, landed=False)
                return
            record.was_pasted = True
            LOGGER.info("Dictation inserted %d characters", len(text))
            self.state = "idle"
            self._maybe_send(fallback)
            self._set_menubar("idle")
            self._remember(text, context.app_name)
            self._insertion = Insertion(text=text, raw=raw, pid=frontmost_pid(), context=context)
            self._keys_wanted = True
            if fallback in ("key", "trial"):
                self._show_raw_key(fallback)
                self._refresh_issues()
            elif fallback:
                self.capsule.show("inserted-raw-net", why=fallback)
            else:
                self.capsule.show("inserted", n=count_units(text), replaced=bool(context.selected_text))
            return

        LOGGER.info(
            "Nowhere to paste (app=%s role=%s, %s); showing the text to copy",
            context.app_name or "unknown",
            context.focused_role or "unknown",
            where,
        )
        self._remember(text, context.app_name)
        set_clipboard_text(text)
        self.state = "idle"
        self._set_menubar("idle")
        self.capsule.show("edit-notarget", text=text)
        self._maybe_send(fallback, landed=False)

    def _maybe_send(self, fallback: str, landed: bool = True) -> None:
        """Press Return after the paste when this dictation was finished with Enter or a double tap."""

        with _SEND_LOCK:
            self._delivered = (getattr(self, "_active_session_id", 0), landed, fallback)
            sid, self._send_session = getattr(self, "_send_session", None), None
        if sid is not None:
            self._send(sid, landed, fallback)

    def _send_when_pasted(self, sid: int) -> None:
        """Send dictation ``sid`` with Return: after its paste, or now if it is already pasted."""

        with _SEND_LOCK:
            delivered = getattr(self, "_delivered", None)
            if delivered is None or delivered[0] != sid:
                LOGGER.info("Double tap: sending once the text is pasted")
                self._send_session = sid
                return
        LOGGER.info("Double tap after the paste: sending now")
        self._send(sid, delivered[1], delivered[2])

    def _send(self, sid: int, landed: bool, fallback: str) -> None:
        if not landed:
            LOGGER.info("Not sending with Return: the text did not land in a field")
        elif sid != getattr(self, "_active_session_id", 0):
            LOGGER.info("Not sending with Return: a new dictation has begun")
        elif fallback:
            LOGGER.info("Not sending with Return: the raw transcript was pasted (%s)", fallback)
        else:
            LOGGER.info("Sending with Return after the paste")
            threading.Timer(SEND_SETTLE_S, self._press_return, args=(sid,)).start()

    def _press_return(self, sid: int) -> None:
        with self._lock:
            if sid == getattr(self, "_active_session_id", 0) and self.state == "idle":
                press_return()
            else:
                LOGGER.info("Not sending with Return: a new dictation has begun")

    def _paste_landed(self, pid: int, before: int) -> bool:
        """Whether the focused field's length changed soon after the Cmd+V."""

        deadline = time.monotonic() + PASTE_CHECK_S
        while True:
            now = focused_text_length(pid)
            if now is None or now != before:
                return True  # changed, or no longer measurable: not evidence of a lost paste
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _keep_recording(self, audio: np.ndarray, sample_rate: int, started: float) -> None:
        """Save this dictation's audio, dropped ones included, when configured to."""

        keep = int(getattr(self.config, "keep_recordings", 0) or 0)
        user_paths = getattr(self.config, "user_paths", None)
        if keep <= 0 or user_paths is None:
            return
        # Named by start time plus the trace.db session id it belongs to.
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(started))
        trace_id = getattr(self, "_last_trace_id", None)
        if trace_id is not None:
            name = f"{name}-{int(trace_id):06d}"
        try:
            keep_recording(user_paths.config_dir / "recordings", name, audio, sample_rate, keep)
        except Exception:
            LOGGER.warning("Could not keep the recording", exc_info=True)

    def _is_current_processing_session(self, session_id: int) -> bool:
        return self.state == "processing" and session_id == getattr(self, "_active_session_id", 0)

    def _prewarm_refiner(self) -> None:
        """Connect to the refinement API ahead of refine, at most every few seconds.

        Called when recording starts, as each stretch is cut, and at the stop:
        with stretches heard while he talks, Whisper is nearly done at the stop,
        so a prewarm only there no longer runs ahead of refine. The client drops
        a connection idle for about 5 s, so one at the start alone expires on a
        long dictation.
        """

        prewarm = getattr(getattr(self, "refiner", None), "prewarm", None)
        now = time.monotonic()
        if prewarm is None or now - getattr(self, "_prewarmed_at", -PREWARM_INTERVAL_S) < PREWARM_INTERVAL_S:
            return
        self._prewarmed_at = now
        threading.Thread(target=prewarm, daemon=True, name="refine-prewarm").start()

    def _should_drop_transcript(self, transcript) -> bool:
        text = str(getattr(transcript, "text", "") or "").strip()
        if not text:
            return True
        language = str(getattr(transcript, "language", "") or "").lower()
        if language and len(text) <= SHORT_FRAGMENT_CHARS:
            confidence = float(getattr(transcript, "confidence", 0.0) or 0.0)
            # A language the user chose in Settings is theirs even if the Mac does not list it.
            spoken = str((getattr(self.asr, "_asr_config", None) or {}).get("language") or "")
            if language not in self._allowed_languages() | {spoken}:
                reason = "language the user does not speak"
            elif language != "zh" and confidence < MIN_SHORT_ENGLISH_CONFIDENCE:
                reason = "low confidence"
            else:
                return False
            LOGGER.info(
                "Dropping short ASR fragment (%s): lang=%s conf=%.2f text=%r", reason, language, confidence, text
            )
            return True
        return False

    def _show_empty_then_idle(self, session_id: int | None = None) -> None:
        """Nothing worth pasting was heard; the capsule says so and goes away by itself."""

        session_id = getattr(self, "_active_session_id", 0) if session_id is None else session_id
        if not self._is_current_processing_session(session_id):
            return
        self.state = "idle"
        self._set_menubar("idle")
        self.capsule.show("empty", device=getattr(self, "_capture_device", "") or "")

    def _copy_last_transcript(self) -> None:
        text = getattr(self, "_copy_fallback_text", "")
        if not text:
            return
        set_clipboard_text(text)
        self.capsule.show("copied")

    def _on_audio_level(self, level: float) -> None:
        if self.state == "recording":
            self.capsule.update_level(level)

    def _call_ui(self, callback, *args, **kwargs) -> None:
        if threading.current_thread() is threading.main_thread():
            callback(*args, **kwargs)
        else:
            AppHelper.callAfter(callback, *args, **kwargs)

    def _log_processing_done(self, future) -> None:
        try:
            future.result()
        except Exception:
            LOGGER.exception("Processing worker crashed")


def _join(parts: list[Transcript]) -> Transcript:
    """The stretches' words in order; language and confidence from the longest one."""

    text = ""
    for part in parts:
        piece = part.text
        if text and piece and _latin(text[-1]) and _latin(piece[0]):
            text += " "
        text += piece
    longest = max(parts, key=lambda part: len(part.text))
    return Transcript(text=text, language=longest.language, confidence=longest.confidence)


def _latin(char: str) -> bool:
    return char.isascii() and char.isalnum()


_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:['’.\-][A-Za-z0-9]+)*")


def count_units(text: str) -> int:
    """字数 the way Chinese editors count it: each Chinese character is one,
    and so is each Latin word."""

    cjk = sum(1 for char in text if brand._is_cjk(char) and unicodedata.category(char).startswith("L"))
    return cjk + len(_LATIN_WORD.findall(text))


OPENAI_BILLING_URL = "https://platform.openai.com/settings/organization/billing/overview"


def _failure_kind(exc: BaseException) -> str:
    """How refinement failed, in the capsule's words: "credit", "badkey", "timeout" or "error"."""

    name = type(exc).__name__.lower()
    code = str(getattr(exc, "code", "") or "")
    if code in {"insufficient_quota", "credit_balance_exhausted"}:
        return "credit"  # the user's own OpenAI account ran out of money
    if name == "authenticationerror" or code == "invalid_api_key":
        return "badkey"
    if "timeout" in name or "timed out" in str(exc).lower():
        return "timeout"
    return "error"


def _eta_text(fraction: float, elapsed: float) -> str:
    """Rough time left for the model download, once there is enough to go on."""

    if fraction < 0.03 or elapsed < 3.0 or fraction >= 1.0:
        return ""
    remaining = elapsed * (1.0 - fraction) / fraction
    if remaining < 60:
        seconds = max(5, int(round(remaining / 5.0)) * 5)
        return t(f"约 {seconds} 秒", f"about {seconds} s")
    minutes = int(round(remaining / 60.0))
    return t(f"约 {minutes} 分钟", f"about {minutes} min")


def api_key_names(config: AppConfig) -> list[str]:
    """Every environment variable an API key is read from: each refinement preset's, and Qwen recognition's."""

    names = [config.refine.api_key_env, cloud_asr.KEY_ENV]
    for preset in preset_names(config.jarvis_config):
        try:
            names.append(refine_config_for(config.jarvis_config, preset).api_key_env)
        except Exception:
            continue
    return list(dict.fromkeys(name for name in names if name))


def configure_logging() -> None:
    """Configure process logging."""

    from logging.handlers import RotatingFileHandler

    level_name = os.environ.get("TYPELESS_LOCAL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        # Runs before load_config, so it is the first thing to touch the config
        # directory and therefore the one that has to carry the old one over.
        log_dir = migrate_legacy_config_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "app.log"
        handlers.append(
            RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5)
        )
    except Exception:
        pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def claim_single_instance(config_dir: Path):
    """Hold a lock on ~/.typlus/app.lock for the life of the process.

    Returns the open lock file, or None when another copy already holds it:
    two copies (the installed app and one run from source) would both answer
    F5 and paste every dictation twice.
    """

    import fcntl  # noqa: PLC0415

    config_dir.mkdir(parents=True, exist_ok=True)
    handle = open(config_dir / "app.lock", "a+")  # noqa: SIM115 - held until exit
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _alert(app, title: str, body: str) -> None:
    """A modal alert before the app is up, for the reasons it will not start."""

    try:
        from AppKit import NSAlert  # noqa: PLC0415

        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(body)
        app.activateIgnoringOtherApps_(True)
        alert.runModal()
    except Exception:
        LOGGER.debug("Could not show the alert %r", title, exc_info=True)


def _say_already_running(app) -> None:
    _alert(
        app,
        t(f"{brand.DISPLAY_NAME}已经在运行", f"{brand.ENGLISH_NAME} is already running"),
        t(
            "另一个言字（可能是从源码运行的那个）已经在响应快捷键。先退出它，再打开这个。",
            "Another copy, perhaps one run from source, is already answering the shortcut. Quit it first, then open this one.",
        ),
    )


def _say_unsupported_mac(app, chip: str) -> None:
    if chip == "rosetta":
        _alert(
            app,
            t(f"{brand.DISPLAY_NAME}正在用 Rosetta 打开", f"{brand.ENGLISH_NAME} is running under Rosetta"),
            t(
                f"语音识别需要直接在 Apple 芯片上运行。在访达里选中{brand.DISPLAY_NAME}，按 ⌘I，取消勾选“使用 Rosetta 打开”，然后再打开。",
                f"Speech recognition has to run natively on Apple silicon. Select {brand.ENGLISH_NAME} in Finder, "
                "press ⌘I, turn off “Open using Rosetta”, then open it again.",
            ),
        )
        return
    _alert(
        app,
        t(f"{brand.DISPLAY_NAME}需要 Apple 芯片的 Mac", f"{brand.ENGLISH_NAME} needs a Mac with Apple silicon"),
        t(
            "语音识别在这台 Mac 上用 MLX 运行，只支持 M1 及更新的芯片，这台 Mac 用的是 Intel 处理器。",
            "Speech recognition runs on this Mac with MLX, which needs an M1 chip or later. This Mac has an Intel processor.",
        ),
    )


_INSTANCE_LOCK = None


def main() -> None:
    """Run the macOS app."""

    global _INSTANCE_LOCK
    configure_logging()
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    config = load_config()
    _INSTANCE_LOCK = claim_single_instance(config.user_paths.config_dir)
    prefs = load_preferences(config.user_paths)
    i18n.use(prefs.ui_language)
    if _INSTANCE_LOCK is None:
        LOGGER.error("Another copy of %s is already running; quitting", brand.ENGLISH_NAME)
        _say_already_running(app)
        return
    chip = reach.machine()
    if chip != "apple":
        LOGGER.error("%s needs Apple silicon; this process runs on %s", brand.ENGLISH_NAME, chip)
        _say_unsupported_mac(app, chip)
        return
    if not prefs.onboarding_done and reach.in_mainland_china():
        # OpenAI does not serve mainland China: a new install there starts on
        # DeepSeek, so the key the guide asks for is one that can work.
        config = adopt_default_preset(config, reach.CHINA_PRESET)
    # Keys saved from Settings live in the login keychain; one still in
    # ~/.typlus/env (or the environment) is used as it is.
    keychain.fill_environ(api_key_names(config))
    trial.ensure_token()
    if not prefs.onboarding_done and reach.in_trial_region() and not os.environ.get(trial.OWN_KEY_ENV):
        # A new US or Canadian Mac with no key starts on the free trial, so the
        # first dictations are refined; elsewhere the guide asks for a key.
        config = adopt_default_preset(config, trial.PRESET)
    coordinator = TypelessLocalApp(config)
    coordinator.start()
    app.run()
