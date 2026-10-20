from __future__ import annotations

from types import SimpleNamespace

from typeless_local import devices


def _fake_portaudio(monkeypatch) -> list[str]:
    calls: list[str] = []
    fake = SimpleNamespace(
        _terminate=lambda: calls.append("terminate"),
        _initialize=lambda: calls.append("initialize"),
    )
    monkeypatch.setattr(devices, "_sounddevice", lambda: fake)
    monkeypatch.setattr(devices, "_last_signature", None)
    return calls


def test_refresh_is_skipped_while_coreaudio_reports_no_change(monkeypatch) -> None:
    calls = _fake_portaudio(monkeypatch)
    signature = [((10, 11), (10,), (11,))]
    monkeypatch.setattr(devices, "hardware_signature", lambda: signature[0])

    assert devices.refresh_if_changed() is True
    assert devices.refresh_if_changed() is False
    signature[0] = ((10, 11, 12), (12,), (11,))  # a mic was plugged in
    assert devices.refresh_if_changed() is True

    assert calls == ["terminate", "initialize"] * 2


def test_refresh_happens_every_time_when_coreaudio_is_unreadable(monkeypatch) -> None:
    calls = _fake_portaudio(monkeypatch)
    monkeypatch.setattr(devices, "hardware_signature", lambda: None)

    assert devices.refresh_if_changed() is True
    assert devices.refresh_if_changed() is True

    assert calls == ["terminate", "initialize"] * 2


def test_hardware_signature_is_none_without_coreaudio(monkeypatch) -> None:
    monkeypatch.setattr(devices, "_load_coreaudio", lambda: None)

    assert devices.hardware_signature() is None


def _fake_coreaudio(monkeypatch, props: dict, devices_with_input=(), names=None) -> None:
    """CoreAudio as a table: (object, selector, scope) -> value."""

    monkeypatch.setattr(devices, "_load_coreaudio", lambda: object())
    monkeypatch.setattr(
        devices, "_read_u32",
        lambda lib, obj, selector, scope=devices._SCOPE_GLOBAL: props.get((obj, selector, scope)),
    )
    monkeypatch.setattr(devices, "_read_object_ids", lambda lib, selector: tuple(sorted({o for o, _, _ in props} - {1})))
    monkeypatch.setattr(devices, "_has_streams", lambda lib, device, scope: device in devices_with_input)
    monkeypatch.setattr(devices, "_device_name", lambda lib, device: (names or {}).get(device, ""))


G, OUT = devices._SCOPE_GLOBAL, devices._SCOPE_OUTPUT
BLUE, BUILT_IN, USB = devices._fourcc("blue"), devices._fourcc("bltn"), devices._fourcc("usb ")


def test_airpods_as_the_default_mic_are_bluetooth_and_the_macs_own_mic_is_found(monkeypatch) -> None:
    props = {(1, "dIn ", G): 40, (40, "tran", G): BLUE, (50, "tran", G): BUILT_IN, (60, "tran", G): BUILT_IN}
    _fake_coreaudio(monkeypatch, props, devices_with_input={40, 60}, names={60: "MacBook Pro Microphone"})

    assert devices.default_input_is_bluetooth() is True
    assert devices.built_in_input() == "MacBook Pro Microphone"  # 50 is the built-in speakers: no input


def test_a_usb_mic_is_not_bluetooth_and_a_mac_mini_has_no_mic_of_its_own(monkeypatch) -> None:
    _fake_coreaudio(monkeypatch, {(1, "dIn ", G): 40, (40, "tran", G): USB, (50, "tran", G): BUILT_IN}, devices_with_input={40})

    assert devices.default_input_is_bluetooth() is False
    assert devices.built_in_input() == ""


def test_headphones_are_bluetooth_output_or_the_headphone_jack(monkeypatch) -> None:
    def output(transport, source=None) -> bool:
        props = {(1, "dOut", G): 70, (70, "tran", G): transport}
        if source:
            props[(70, "ssrc", OUT)] = devices._fourcc(source)
        _fake_coreaudio(monkeypatch, props)
        return devices.output_is_headphones()

    assert output(BLUE) is True
    assert output(devices._fourcc("blea")) is True
    assert output(BUILT_IN, "hdpn") is True
    assert output(BUILT_IN, "ispk") is False
    assert output(USB) is False


def test_nothing_is_headphones_or_bluetooth_without_coreaudio(monkeypatch) -> None:
    monkeypatch.setattr(devices, "_load_coreaudio", lambda: None)

    assert devices.output_is_headphones() is False
    assert devices.default_input_is_bluetooth() is False
    assert devices.built_in_input() == ""
