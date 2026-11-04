"""Silencing the speakers while recording, including through a Multi-Output Device."""

from __future__ import annotations

import threading
from typing import Any

from typeless_local import devices


class Ducker:
    """The system ducker, plus muting each device inside an aggregate output.

    ``set volume`` can't lower a Multi-Output Device (it has no volume of its
    own), so when the system ducker finds no volume control, the speakers and
    displays it plays through are muted one by one and put back afterwards.
    """

    def __init__(self, system: Any, parts: Any = devices) -> None:  # noqa: ANN401
        self._system = system
        self._parts = parts
        self._lock = threading.Lock()
        self._changed: list[tuple] = []

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._system, "enabled", True))

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._system.enabled = value

    @property
    def unsupported(self) -> bool:
        return bool(getattr(self._system, "unsupported", False))

    def duck(self) -> bool:
        if self._system.duck():
            return True
        if not (self.enabled and self.unsupported):
            return False
        with self._lock:
            if not self._changed:
                self._changed = self._parts.silence_output_parts()
            return bool(self._changed)

    def restore(self) -> None:
        self._system.restore()
        self._restore_parts()

    def restore_all(self) -> None:
        self._system.restore_all()
        self._restore_parts()

    def _restore_parts(self) -> None:
        with self._lock:
            changed, self._changed = self._changed, []
        if changed:
            self._parts.restore_output_parts(changed)
