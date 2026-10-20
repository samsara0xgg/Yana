"""Which languages this user can plausibly be speaking.

With the spoken language on Automatic, Whisper detects it per clip, and on a
clip of a stray word or two it sometimes names a language nobody here speaks
(Icelandic, Korean). The allowed set narrows that guess to the Mac's own
languages, the enabled keyboards, and what past long dictations were heard as.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Iterable

LOGGER = logging.getLogger(__name__)

# Below this a clip gives Whisper too little to detect a language from.
SHORT_CLIP_S = 4.0
# A long dictation teaches its language only when Whisper was this sure of a
# transcript at least this long.
LEARN_MIN_CONFIDENCE = 0.6
LEARN_MIN_CHARS = 8
MAX_HEARD_LANGUAGES = 8

_CARBON = "/System/Library/Frameworks/Carbon.framework/Carbon"
_CORE_FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_system: frozenset[str] | None = None


def base_code(tag: str) -> str:
    """"zh-Hans-CN" -> "zh"; "en_US" -> "en"."""

    return str(tag).replace("_", "-").split("-")[0].lower()


def whisper_codes(tags: Iterable[str]) -> set[str]:
    """The base codes among ``tags`` that Whisper has a model language for."""

    from mlx_whisper.tokenizer import LANGUAGES  # noqa: PLC0415

    return {code for code in map(base_code, tags) if code in LANGUAGES}


def _input_source_languages() -> list[str]:
    """Languages of the enabled keyboards and input sources (Carbon TIS, main thread only)."""

    import objc  # noqa: PLC0415

    carbon = ctypes.cdll.LoadLibrary(_CARBON)
    cf = ctypes.cdll.LoadLibrary(_CORE_FOUNDATION)
    carbon.TISCreateInputSourceList.restype = ctypes.c_void_p
    carbon.TISCreateInputSourceList.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    carbon.TISGetInputSourceProperty.restype = ctypes.c_void_p
    carbon.TISGetInputSourceProperty.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
    cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    key = ctypes.c_void_p.in_dll(carbon, "kTISPropertyInputSourceLanguages").value
    listing = carbon.TISCreateInputSourceList(None, False)  # enabled sources only
    if not listing:
        return []
    found: list[str] = []
    try:
        for index in range(cf.CFArrayGetCount(listing)):
            langs = carbon.TISGetInputSourceProperty(cf.CFArrayGetValueAtIndex(listing, index), key)
            # A CFArray of CFStrings, toll-free bridged to NSArray. Only the first is the
            # keyboard's own language: the ABC layout lists every Latin-script one (~100).
            items = list(objc.objc_object(c_void_p=langs)) if langs else []
            found.extend(str(item) for item in items[:1])
    finally:
        cf.CFRelease(listing)
    return found


def load_system() -> frozenset[str]:
    """Read the Mac's languages once and keep them. Main thread only (TIS); never on the event tap."""

    global _system
    if _system is not None:
        return _system
    tags: list[str] = []
    try:
        from Foundation import NSLocale  # noqa: PLC0415

        tags.extend(str(tag) for tag in NSLocale.preferredLanguages())
    except Exception:
        LOGGER.debug("Could not read the Mac's preferred languages", exc_info=True)
    try:
        tags.extend(_input_source_languages())
    except Exception:
        LOGGER.debug("Could not read the input sources' languages", exc_info=True)
    _system = frozenset(whisper_codes(tags))
    LOGGER.info("System languages for Automatic recognition: %s", sorted(_system))
    return _system


def known() -> bool:
    """Whether the Mac's own languages have been read yet."""

    return _system is not None


def allowed(heard: Iterable[str] = ()) -> set[str]:
    """System languages, learned ones, and English. Until the system set is read, just the rest."""

    return set(_system or ()) | whisper_codes(heard) | {"en"}


def pick(probs: dict[str, float], allowed_set: set[str]) -> str | None:
    """The most probable language of ``probs`` that is allowed; None if none is."""

    candidates = {lang: p for lang, p in probs.items() if lang in allowed_set}
    return max(candidates, key=candidates.get) if candidates else None
