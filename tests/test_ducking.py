from __future__ import annotations

from types import SimpleNamespace

from typeless_local.ducking import Ducker


class _System:
    def __init__(self, ducks: bool, unsupported: bool = False) -> None:
        self.ducks, self.unsupported, self.enabled = ducks, unsupported, True
        self.calls = []

    def duck(self) -> bool:
        self.calls.append("duck")
        return self.ducks

    def restore(self) -> None:
        self.calls.append("restore")

    def restore_all(self) -> None:
        self.calls.append("restore_all")


def _parts(changed):
    log = []
    return log, SimpleNamespace(
        silence_output_parts=lambda: log.append("silence") or list(changed),
        restore_output_parts=lambda items: log.append(("restore", items)),
    )


def test_a_multi_output_device_is_silenced_through_the_devices_inside_it() -> None:
    log, parts = _parts([("speakers", "mute")])
    ducker = Ducker(_System(ducks=False, unsupported=True), parts)

    assert ducker.duck() is True
    ducker.restore_all()
    ducker.restore_all()  # a second restore puts nothing back twice

    assert log == ["silence", ("restore", [("speakers", "mute")])]


def test_an_output_with_its_own_volume_is_left_to_the_system_ducker() -> None:
    log, parts = _parts([("speakers", "mute")])
    system = _System(ducks=True)
    ducker = Ducker(system, parts)

    assert ducker.duck() is True
    ducker.restore_all()

    assert log == [] and system.calls == ["duck", "restore_all"]


def test_with_ducking_off_nothing_inside_is_touched() -> None:
    log, parts = _parts([("speakers", "mute")])
    system = _System(ducks=False, unsupported=True)
    ducker = Ducker(system, parts)
    ducker.enabled = False

    assert ducker.duck() is False
    assert log == [] and system.enabled is False


def test_nothing_inside_that_can_be_muted_counts_as_not_ducked() -> None:
    log, parts = _parts([])
    ducker = Ducker(_System(ducks=False, unsupported=True), parts)

    assert ducker.duck() is False
