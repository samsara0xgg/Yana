"""What the capsule shows, and when it goes away by itself.

The overlay only draws. This decides how long each state stays: a state that
reports something finished goes away after a moment, the pointer resting on the
capsule holds it, and every new state supersedes whatever timer the previous
one left running, so a late timer can never take down a newer state.

Safe to call from any thread. The overlay itself is only touched through
``call_ui``, which runs it on the main thread, and the capsule never calls back
into the app, so the app may call it while holding its own lock.
"""

from __future__ import annotations

from functools import partial
import logging
import threading
import time
from typing import Callable

LOGGER = logging.getLogger(__name__)

# Seconds a state stays up by itself. Not listed: stays until replaced.
DISMISS_AFTER: dict[str, float] = {
    "inserted-raw-net": 8.0,
    "inserted-raw-key": 8.0,
    "inserted-raw-trial": 8.0,
    "inserted-unsure": 8.0,
    "empty": 2.4,
    "cancelled": 0.8,
    "undone": 1.2,
    "replaced": 1.6,
    "copied": 1.4,
    "mic": 6.0,
    "error": 8.0,
    "notice": 2.6,
    "ready": 6.0,
    "perm": 12.0,
}
# The English practice card stays for the time it takes to read: see english_dismiss_s.
ENGLISH_BASE_S, ENGLISH_PER_CHAR_S, ENGLISH_MAX_S = 6.0, 0.06, 20.0
INSERTED_STATES = frozenset({"inserted", "inserted-raw-net", "inserted-raw-key", "inserted-raw-trial"})
# Once the pointer leaves, a state it was holding gets at least this long.
RESUME_MIN_S = 1.2
_DEFAULT = object()


class Capsule:
    def __init__(
        self,
        overlay,
        call_ui: Callable,
        *,
        timer=threading.Timer,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.overlay = overlay
        self._call_ui = call_ui
        self._timer_factory = timer
        self._clock = clock
        self._lock = threading.RLock()
        self.state = "hidden"
        self.data: dict = {}
        # "inserted" follows the user's setting; see preferences.dismiss_seconds.
        self.inserted_dismiss_s = 4.0
        self._seq = 0
        self._timer = None
        self._due = 0.0
        self._remaining: float | None = None
        self._hover = False

    def show(self, state: str, *, dismiss=_DEFAULT, **data) -> None:
        """Show ``state``; ``dismiss`` overrides how long it stays (None: until replaced)."""

        with self._lock:
            self._seq += 1
            self.state = state
            self.data = dict(data)
            self._cancel_timer()
            delay = self._delay_for(state) if dismiss is _DEFAULT else dismiss
            self._remaining = delay
            if delay is not None and not self._hover:
                self._start_timer(delay)
            # Drawn under the lock so the overlay sees states in the order they were set.
            self._draw(self.overlay and self.overlay.show, state, **data)

    def hide(self) -> None:
        with self._lock:
            self._seq += 1
            self.state = "hidden"
            self.data = {}
            self._cancel_timer()
            self._remaining = None
            self._draw(self.overlay and self.overlay.hide)

    def hide_if(self, *states: str) -> bool:
        """Hide only while one of ``states`` is showing; True if it did."""

        with self._lock:
            if self.state not in states:
                return False
            self.hide()
            return True

    def set_hover(self, on: bool) -> None:
        """The pointer rests on the capsule: whatever was about to go away waits."""

        with self._lock:
            on = bool(on)
            if on == self._hover:
                return
            self._hover = on
            if on:
                if self._timer is not None:
                    self._remaining = max(0.0, self._due - self._clock())
                    self._cancel_timer()
            elif self._remaining is not None and self.state != "hidden":
                self._start_timer(max(self._remaining, RESUME_MIN_S))

    # ------------------------------------------------------ pass-throughs

    def end_edit(self) -> None:
        self._draw(self.overlay and self.overlay.end_edit)

    def update_level(self, level: float) -> None:
        self._draw(self.overlay and self.overlay.update_level, level)

    def set_handle(self, on: bool) -> None:
        self._draw(self.overlay and self.overlay.set_handle, bool(on))

    def set_anchor(self, mode: str, caret=None) -> None:
        self._draw(self.overlay and self.overlay.set_anchor, mode, caret)

    # ----------------------------------------------------------- internals

    def _delay_for(self, state: str) -> float | None:
        if state == "english":
            n = len(self.data.get("en", "")) + sum(len(p.get("en", "")) + len(p.get("zh", "")) for p in self.data.get("phrases", []))
            return min(ENGLISH_MAX_S, ENGLISH_BASE_S + ENGLISH_PER_CHAR_S * n)
        if state == "inserted":
            return float(self.inserted_dismiss_s)
        return DISMISS_AFTER.get(state)

    def _start_timer(self, delay: float) -> None:
        self._due = self._clock() + delay
        timer = self._timer_factory(delay, partial(self._expire, self._seq))
        timer.daemon = True
        timer.start()
        self._timer = timer

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _expire(self, seq: int) -> None:
        with self._lock:
            if seq != self._seq or self.state == "hidden":
                return
            self._timer = None
            self.hide()

    def _draw(self, method, *args, **kwargs) -> None:
        if method:
            self._call_ui(method, *args, **kwargs)
