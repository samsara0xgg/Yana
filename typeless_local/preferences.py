"""Preferences set from the Settings window.

They live under ``ui:`` in ~/.typlus/config.yaml, next to the engine's own
sections, so one file still holds everything a user has changed. Settings that
belong to an existing section (the recognition language, the refine preset,
the input device, ducking) are written there instead, where the engine already
reads them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import logging
from typing import Any

from typeless_local import i18n, reach
from typeless_local.config import UserPaths, load_yaml, save_user_setting

LOGGER = logging.getLogger(__name__)

SECTION = "ui"
# Allowed values for the settings that are a choice rather than a switch.
CHOICES: dict[str, tuple[Any, ...]] = {
    "capsule_position": ("bottom", "caret"),
    "dismiss_seconds": (2.0, 4.0, 6.0, 10.0),
    "sounds": ("off", "start_end"),
    "max_minutes": (5, 9, 15),
    # 0 keeps everything. Nothing is ever deleted unless the user picks a limit.
    "history_days": (30, 90, 365, 0),
    # "auto" speaks the Mac's own language: Chinese on a Chinese system, English otherwise.
    "ui_language": i18n.CHOICES,
    # Where the speech model downloads from; "auto" is the mirror in mainland China.
    "model_source": reach.MODEL_SOURCES,
    # Who hears the recording: Whisper on this Mac, or Qwen in Alibaba's cloud while
    # he talks (verbatim, or the voice-input model that tidies as it goes).
    "asr_engine": ("local", "qwen", "qwen_message"),
}


@dataclass(frozen=True)
class Preferences:
    capsule_position: str = "bottom"
    show_handle: bool = True
    dismiss_seconds: float = 4.0
    sounds: str = "off"
    max_minutes: int = 15
    refine: bool = True
    rewrite_selection: bool = True
    show_ducked: bool = False
    save_history: bool = True
    history_days: int = 0
    send_window_title: bool = True
    # After a paste, keep what the text finally went out as (sent_text.py).
    save_sent_text: bool = True
    # Off by default: it costs tokens and a few AX reads, and has not yet been
    # shown to help. When off, the text is not even read.
    send_before_text: bool = False
    # Refinement also writes an English version, shown in a card after the paste.
    english_practice: bool = False
    # Once a day: dictation and character counts and trial spend, never text (stats.py).
    send_usage_stats: bool = True
    # F5 is also the system dictation key, so it is left to macOS unless asked
    # for; right Command is the shortcut. Installs from before 0.4.0 keep F5 on.
    f5_hotkey: bool = False
    # Off for people whose right Command already switches input sources; then
    # F5 is the shortcut (Settings keeps at least one of the two on).
    right_command_hotkey: bool = True
    # The trial server said this Mac's free refinements are spent; it is not asked again.
    trial_used_up: bool = False
    onboarding_done: bool = False
    ui_language: str = "auto"
    model_source: str = "auto"
    asr_engine: str = "local"
    # Languages heard in confident long dictations; with the spoken language on
    # Automatic they are allowed on short clips (languages.py). Not in Settings.
    heard_languages: list[str] = field(default_factory=list)
    # Players whose play/pause showed no change in their output (NetEase keeps it
    # open while paused); the key is never pressed for them. Not in Settings.
    unclear_media_apps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULTS = Preferences()
_TYPES = {field.name: type(getattr(_DEFAULTS, field.name)) for field in fields(Preferences)}


def coerce(key: str, value: Any) -> Any:
    """``value`` as the type ``key`` holds; ValueError when it cannot be."""

    if key not in _TYPES:
        raise ValueError(f"unknown preference {key!r}")
    kind = _TYPES[key]
    if kind is bool:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off", ""}:
                return False
            raise ValueError(f"{key}: not a switch value: {value!r}")
        return bool(value)
    try:
        coerced = kind(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}: {value!r} is not a {kind.__name__}") from exc
    allowed = CHOICES.get(key)
    if allowed is not None and coerced not in allowed:
        raise ValueError(f"{key}: {coerced!r} is not one of {allowed}")
    return coerced


def _read_section(user_paths: UserPaths) -> dict[str, Any]:
    path = user_paths.user_config_path
    if not path.exists():
        return {}
    try:
        section = load_yaml(path).get(SECTION) or {}
    except Exception:
        LOGGER.warning("Could not read %s; using default preferences", path, exc_info=True)
        return {}
    return section if isinstance(section, dict) else {}


def load_preferences(user_paths: UserPaths | None) -> Preferences:
    """The saved preferences; a value that doesn't fit falls back to its default."""

    if user_paths is None:
        return _DEFAULTS
    values: dict[str, Any] = {}
    section = _read_section(user_paths)
    if section.get("onboarding_done") and "f5_hotkey" not in section:
        values["f5_hotkey"] = True  # set up before right Command took over; F5 is what they know
    for key, raw in section.items():
        try:
            values[key] = coerce(key, raw)
        except ValueError:
            LOGGER.warning("Ignoring preference %s=%r", key, raw)
    return replace(_DEFAULTS, **values)


def save_preference(user_paths: UserPaths | None, prefs: Preferences, key: str, value: Any) -> Preferences:
    """Validate, persist and return the updated preferences."""

    coerced = coerce(key, value)
    updated = replace(prefs, **{key: coerced})
    if user_paths is not None:
        save_user_setting(user_paths, SECTION, key, coerced)
    return updated
