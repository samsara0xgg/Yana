"""Audio device enumeration, system output switching, and hardware-AEC pairing.

Ducking exists because the microphone would otherwise pick up whatever the
speakers are playing. A capture device with hardware echo cancellation removes
that reason, but only while it is fed the same signal the speakers get, which
means the pairing of input and output device is what decides whether ducking is
still needed. ``has_hardware_aec`` answers exactly that question.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import shutil
import subprocess
from typing import Any

LOGGER = logging.getLogger(__name__)

SWITCH_AUDIO_SOURCE = "SwitchAudioSource"
SYSTEM_DEFAULT = ""
# Launched from Finder the app inherits no shell PATH, so a Homebrew install is
# invisible to shutil.which. Same reason API keys live in ~/.typlus/env.
_SWITCH_FALLBACK_PATHS = (
    "/opt/homebrew/bin/SwitchAudioSource",
    "/usr/local/bin/SwitchAudioSource",
)


def _switch_binary() -> str | None:
    found = shutil.which(SWITCH_AUDIO_SOURCE)
    if found:
        return found
    for path in _SWITCH_FALLBACK_PATHS:
        if os.access(path, os.X_OK):
            return path
    return None


def _sounddevice() -> Any:  # noqa: ANN401 - module handle
    import sounddevice as sd  # noqa: PLC0415

    return sd


def _names(kind: str) -> list[str]:
    """Device names exposing at least one channel of ``kind``, in system order."""

    key = f"max_{kind}_channels"
    try:
        devices = _sounddevice().query_devices()
    except Exception:
        LOGGER.warning("Unable to enumerate audio devices", exc_info=True)
        return []
    out: list[str] = []
    seen: set[str] = set()
    for device in devices:
        name = str(device.get("name") or "").strip()
        if not name or name in seen or int(device.get(key, 0) or 0) <= 0:
            continue
        seen.add(name)
        out.append(name)
    return out


def list_input_devices() -> list[str]:
    return _names("input")


def list_output_devices() -> list[str]:
    return _names("output")


def _default_name(slot: int) -> str:
    """Name of the current system default device; "" when it cannot be read."""

    try:
        sd = _sounddevice()
        index = sd.default.device[slot]
        if index is None or int(index) < 0:
            return ""
        return str(sd.query_devices(int(index)).get("name") or "").strip()
    except Exception:
        LOGGER.debug("Unable to read default audio device %d", slot, exc_info=True)
        return ""


def current_input_device() -> str:
    return _default_name(0)


def current_output_device() -> str:
    return _default_name(1)


class _PropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


_SYSTEM_OBJECT = 1  # kAudioObjectSystemObject
_SCOPE_GLOBAL = _fourcc("glob")
_ELEMENT_MAIN = 0
_HARDWARE_SELECTORS = (
    _fourcc("dev#"),  # kAudioHardwarePropertyDevices
    _fourcc("dIn "),  # kAudioHardwarePropertyDefaultInputDevice
    _fourcc("dOut"),  # kAudioHardwarePropertyDefaultOutputDevice
)
_coreaudio: Any = None
_coreaudio_loaded = False
_last_signature: tuple | None = None


def _load_coreaudio() -> Any:  # noqa: ANN401 - ctypes library handle
    global _coreaudio, _coreaudio_loaded
    if _coreaudio_loaded:
        return _coreaudio
    _coreaudio_loaded = True
    try:
        path = ctypes.util.find_library("CoreAudio")
        if not path:
            return None
        lib = ctypes.CDLL(path)
        address = ctypes.POINTER(_PropertyAddress)
        size = ctypes.POINTER(ctypes.c_uint32)
        lib.AudioObjectGetPropertyDataSize.argtypes = [
            ctypes.c_uint32, address, ctypes.c_uint32, ctypes.c_void_p, size,
        ]
        lib.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
        lib.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32, address, ctypes.c_uint32, ctypes.c_void_p, size, ctypes.c_void_p,
        ]
        lib.AudioObjectGetPropertyData.restype = ctypes.c_int32
        _coreaudio = lib
    except Exception:
        LOGGER.debug("CoreAudio unavailable; devices re-read on every recording", exc_info=True)
    return _coreaudio


def _read_object_ids(lib: Any, selector: int) -> tuple[int, ...] | None:  # noqa: ANN401
    address = _PropertyAddress(selector, _SCOPE_GLOBAL, _ELEMENT_MAIN)
    size = ctypes.c_uint32(0)
    if lib.AudioObjectGetPropertyDataSize(
        _SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(size)
    ):
        return None
    buffer = (ctypes.c_uint32 * max(1, size.value // 4))()
    if lib.AudioObjectGetPropertyData(
        _SYSTEM_OBJECT, ctypes.byref(address), 0, None, ctypes.byref(size), buffer
    ):
        return None
    return tuple(buffer[: size.value // 4])


def hardware_signature() -> tuple | None:
    """CoreAudio's device IDs and default input/output, or None if unreadable.

    A few microseconds to read, unlike re-initialising PortAudio, so it can sit
    on the hotkey path and decide whether that heavier re-read is needed.
    """

    lib = _load_coreaudio()
    if lib is None:
        return None
    try:
        parts = tuple(_read_object_ids(lib, selector) for selector in _HARDWARE_SELECTORS)
    except Exception:
        LOGGER.debug("Unable to read CoreAudio devices", exc_info=True)
        return None
    if any(part is None for part in parts):
        return None
    return parts


def _read_u32(lib: Any, obj: int, selector: str, scope: int = _SCOPE_GLOBAL) -> int | None:  # noqa: ANN401
    address = _PropertyAddress(_fourcc(selector), scope, _ELEMENT_MAIN)
    value = ctypes.c_uint32(0)
    size = ctypes.c_uint32(4)
    if lib.AudioObjectGetPropertyData(obj, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(value)):
        return None
    return value.value


_SCOPE_INPUT = _fourcc("inpt")
_SCOPE_OUTPUT = _fourcc("outp")
_BLUETOOTH = frozenset({_fourcc("blue"), _fourcc("blea")})  # kAudioDeviceTransportTypeBluetooth(LE)
_BUILT_IN = _fourcc("bltn")
_HEADPHONE_JACK = _fourcc("hdpn")  # the built-in output's data source with headphones plugged in
_UTF8 = 0x08000100  # kCFStringEncodingUTF8
_corefoundation: Any = None


def _has_streams(lib: Any, device: int, scope: int) -> bool:  # noqa: ANN401
    address = _PropertyAddress(_fourcc("stm#"), scope, _ELEMENT_MAIN)
    size = ctypes.c_uint32(0)
    if lib.AudioObjectGetPropertyDataSize(device, ctypes.byref(address), 0, None, ctypes.byref(size)):
        return False
    return size.value > 0


def _device_name(lib: Any, device: int) -> str:  # noqa: ANN401
    """The name PortAudio lists the device under (kAudioObjectPropertyName)."""

    global _corefoundation
    address = _PropertyAddress(_fourcc("lnam"), _SCOPE_GLOBAL, _ELEMENT_MAIN)
    ref = ctypes.c_void_p()
    size = ctypes.c_uint32(ctypes.sizeof(ref))
    if lib.AudioObjectGetPropertyData(device, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(ref)):
        return ""
    if not ref.value:
        return ""
    if _corefoundation is None:
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        _corefoundation = cf
    buffer = ctypes.create_string_buffer(512)
    try:
        ok = _corefoundation.CFStringGetCString(ref, buffer, len(buffer), _UTF8)
    finally:
        _corefoundation.CFRelease(ref)
    return buffer.value.decode("utf-8", "replace").strip() if ok else ""


def default_input_is_bluetooth() -> bool:
    """Whether the system's default mic belongs to a Bluetooth headset.

    Opening that mic switches the headset from music to call mode: whatever is
    playing cuts out, then comes back at phone quality until the mic closes.
    """

    lib = _load_coreaudio()
    if lib is None:
        return False
    try:
        device = _read_u32(lib, _SYSTEM_OBJECT, "dIn ")
        return bool(device) and _read_u32(lib, device, "tran") in _BLUETOOTH
    except Exception:
        LOGGER.debug("Unable to read the default input's transport", exc_info=True)
        return False


def built_in_input() -> str:
    """The Mac's own microphone by name, or "" on a Mac without one."""

    lib = _load_coreaudio()
    if lib is None:
        return ""
    try:
        for device in _read_object_ids(lib, _fourcc("dev#")) or ():
            if _read_u32(lib, device, "tran") == _BUILT_IN and _has_streams(lib, device, _SCOPE_INPUT):
                return _device_name(lib, device)
    except Exception:
        LOGGER.debug("Unable to find the built-in microphone", exc_info=True)
    return ""


def output_is_headphones() -> bool:
    """Whether the system plays into someone's ears, where no mic can hear it.

    Bluetooth output counts as headphones: CoreAudio can't tell a Bluetooth
    speaker from a headset, and on a Mac it is nearly always a headset. Wired
    ones show up as the built-in output with its headphone data source.
    """

    lib = _load_coreaudio()
    if lib is None:
        return False
    try:
        device = _read_u32(lib, _SYSTEM_OBJECT, "dOut")
        if not device:
            return False
        transport = _read_u32(lib, device, "tran")
        if transport in _BLUETOOTH:
            return True
        return transport == _BUILT_IN and _read_u32(lib, device, "ssrc", _SCOPE_OUTPUT) == _HEADPHONE_JACK
    except Exception:
        LOGGER.debug("Unable to read the default output's transport", exc_info=True)
        return False


# Calls hold output open all through a meeting and ignore the play/pause key,
# so they say nothing about whether music is playing.
CALL_APPS = frozenset({
    "us.zoom.xos", "com.tencent.xinWeChat", "com.microsoft.teams", "com.microsoft.teams2",
    "com.hnc.Discord", "com.apple.FaceTime", "com.skype.skype", "com.cisco.webexmeetingsapp",
    "com.webex.meetingmanager", "com.tencent.meeting", "com.alibaba.DingTalkMac", "com.electron.lark",
    "com.bytedance.macos.feishu", "com.tinyspeck.slackmacgap",
})


def bundle_id(pid: int) -> str:
    """The app's bundle id, or "" when it has none or can't be read."""

    try:
        from AppKit import NSRunningApplication

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        return str(app.bundleIdentifier() or "") if app is not None else ""
    except Exception:
        return ""


def is_call_app(pid: int) -> bool:
    return bundle_id(pid) in CALL_APPS


def _responsible_pid(pid: int) -> int:
    """The app a helper process works for (Chrome's audio service, Safari's WebKit GPU process)."""

    global _responsible
    if _responsible is None:
        try:
            fn = ctypes.CDLL("/usr/lib/libSystem.B.dylib").responsibility_get_pid_responsible_for_pid
            fn.restype, fn.argtypes = ctypes.c_int, [ctypes.c_int]
            _responsible = fn
        except Exception:
            _responsible = lambda pid: pid  # noqa: E731
    try:
        return int(_responsible(pid)) or pid
    except Exception:
        return pid


_responsible = None


def playing_apps() -> set[int]:
    """The Dock apps other than this one sending audio out right now, by pid.

    CoreAudio lists which processes are running output. Background daemons are
    skipped (Jarvis holds a silent output stream open all day) by requiring
    the process, or the app responsible for a helper such as Chrome's audio
    service or Safari's WebKit GPU process, to be a regular app. A call (Zoom, WeChat) counts too: it holds output open
    whether or not anyone is talking.
    """

    found: set[int] = set()
    lib = _load_coreaudio()
    if lib is None:
        return found
    try:
        from AppKit import NSApplicationActivationPolicyRegular, NSRunningApplication

        def regular(pid: int) -> bool:
            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            return app is not None and app.activationPolicy() == NSApplicationActivationPolicyRegular

        for process in _read_object_ids(lib, _fourcc("prs#")) or ():
            pid = _read_u32(lib, process, "ppid")
            if not pid or pid == os.getpid() or not _read_u32(lib, process, "piro"):
                continue
            if regular(pid):
                found.add(pid)
                continue
            owner = _responsible_pid(pid)
            if owner != pid and owner != os.getpid() and regular(owner):
                found.add(owner)
    except Exception:
        LOGGER.debug("Unable to read which apps are playing", exc_info=True)
    return found


def refresh() -> None:
    """Re-read the hardware list so a mic plugged or pulled since launch is seen.

    PortAudio enumerates once when it initialises, so a long-running process
    keeps offering a device that has since been unplugged and then fails to
    open it with an internal error. Only safe while no stream is open.
    """

    global _last_signature
    sd = _sounddevice()
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        LOGGER.warning("Unable to refresh the audio device list", exc_info=True)
        return
    _last_signature = hardware_signature()


def refresh_if_changed() -> bool:
    """``refresh()`` only when CoreAudio's devices or defaults have changed.

    Re-initialising PortAudio on every recording put a full device enumeration
    between the hotkey and the microphone opening. Where the change can't be
    detected, it refreshes every time, as before. Returns whether it refreshed.
    """

    signature = hardware_signature()
    if signature is not None and signature == _last_signature:
        return False
    refresh()
    return True


def prime_input(index: int | None, sample_rate: int) -> None:
    """Open and close one input stream, so the first recording doesn't pay for it.

    The first stream a launch opened took about 2 s (every first F5 after a
    launch in the log, later ones about 0.05 s), and those words were lost.
    """

    stream = _sounddevice().InputStream(samplerate=sample_rate, channels=1, dtype="float32", device=index)
    stream.close()


_said_missing: set[str] = set()


def resolve_input_index(name: str) -> int | None:
    """sounddevice index for ``name``; None means "let the system decide"."""

    wanted = (name or "").strip()
    if not wanted:
        return None
    try:
        devices = _sounddevice().query_devices()
    except Exception:
        LOGGER.warning("Unable to resolve input device %r", wanted, exc_info=True)
        return None
    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0) or 0) <= 0:
            continue
        if str(device.get("name") or "").strip() == wanted:
            return index
    if wanted not in _said_missing:  # asked on every recording: say it once per device
        _said_missing.add(wanted)
        LOGGER.warning("Input device %r not present; falling back to system default", wanted)
    return None


def can_switch_output() -> bool:
    return _switch_binary() is not None


def set_output_device(name: str) -> bool:
    """Point the system default output at ``name``. False when it did not happen."""

    wanted = (name or "").strip()
    if not wanted:
        return False
    binary = _switch_binary()
    if binary is None:
        LOGGER.warning("%s not found; cannot switch output", SWITCH_AUDIO_SOURCE)
        return False
    try:
        subprocess.run(
            [binary, "-s", wanted, "-t", "output"],
            check=True,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        LOGGER.warning("Failed to switch system output to %r", wanted, exc_info=True)
        return False
    LOGGER.info("System output switched to %s", wanted)
    return True


def has_hardware_aec(input_name: str, output_name: str, pairs: list[Any]) -> bool:
    """Whether this input/output pairing cancels the speakers on its own."""

    if not input_name or not output_name:
        return False
    for pair in pairs or []:
        if not isinstance(pair, dict):
            continue
        if (
            str(pair.get("input") or "").strip() == input_name
            and str(pair.get("output") or "").strip() == output_name
        ):
            return True
    return False
