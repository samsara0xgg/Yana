"""The capsule's window: a transparent, non-activating panel near the bottom of the screen.

The page in ``web/overlay.html`` draws the capsule and animates it. This module
owns the window around it and follows the page: every frame the page reports
the rectangles it drew, native glass views are moved under them, and the panel
takes mouse events only while the pointer is over one of them, so the rest of
the screen stays clickable through it.

The panel never becomes key, except while a card needs the keyboard: a click on
a button must leave the keyboard with the app the text went to, or a ⌘Z sent to
undo a paste would land here instead.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import AppKit
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSEvent,
    NSFloatingWindowLevel,
    NSMakePoint,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
    NSWorkspace,
)
from Foundation import NSObject, NSRunLoop, NSRunLoopCommonModes, NSTimer
from WebKit import WKWebView
import objc

from typeless_local import i18n
from typeless_local.glass import FlippedView, GlassLayer
from typeless_local.webview import WebPage, accessibility_env, announce

LOGGER = logging.getLogger(__name__)

ActionCallback = Callable[[str, dict], None]
HoverCallback = Callable[[bool], None]

PANEL_WIDTH = 720
PANEL_HEIGHT = 520
# While the pointer is over the capsule, how often it is re-read: WebKit gets
# no mouse-moved events in a window that is not key, so hover is fed by hand.
POINTER_INTERVAL = 1 / 30
# Which display the pointer is on is checked at most this often.
FOLLOW_INTERVAL = 0.1
# The capsule under (or over) the caret: gap to the caret, and room the page
# keeps between the capsule and the panel's edge for the glass's shadow.
CARET_GAP = 8.0
CARET_PAD = 16.0
# Height a card can grow to; the side of the caret with less room is not used.
CARET_ROOM = 280.0
# What the page can draw (web/overlay.js, view()) and the actions it posts back.
STATES = frozenset({
    "starting", "rec", "transcribing", "refining",
    "inserted", "inserted-raw-net", "inserted-raw-key", "inserted-raw-trial", "inserted-unsure",
    "edit-notarget", "edit-modify",
    "empty", "mic", "download", "cancelled", "undone", "replaced", "copied",
    "perm", "notice", "ready", "error", "english",
})
CARD_STATES = frozenset({"edit-notarget", "edit-modify"})
ACTIONS = frozenset({
    "primary", "cancel", "finish", "extend",                # idle handle, recording
    "undo", "edit", "rerefine", "setkey", "billing", "input", "micperm", "perm", "log", "restart",  # buds
    "close", "done", "replace", "useen", "copyen",           # cards
})

_A11Y_CHANGED = getattr(
    AppKit,
    "NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification",
    "NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification",
)
_MOUSE_MOVED_MASK = (1 << 5) | (1 << 6) | (1 << 7)  # moved, left and right dragged


def _screen_for_point(point, screens):
    """The screen whose frame contains ``point``, or ``None`` if it is on none.

    ``frame`` is used rather than ``visibleFrame`` so a pointer over the menu
    bar or the Dock still counts as being on that display.
    """

    for screen in screens:
        frame = screen.frame()
        if (
            frame.origin.x <= point.x < frame.origin.x + frame.size.width
            and frame.origin.y <= point.y < frame.origin.y + frame.size.height
        ):
            return screen
    return None


def hit_shape(shapes: list[dict], x: float, y: float) -> str:
    """The id of the clickable shape under page point (x, y), or "".

    Shapes are rounded rectangles in page coordinates (origin top-left), as the
    page reports them; the corners outside the rounding are not the capsule.
    """

    for shape in reversed(shapes):
        if not shape.get("hit"):
            continue
        left, top = float(shape["x"]), float(shape["y"])
        width, height = float(shape["w"]), float(shape["h"])
        if not (left <= x <= left + width and top <= y <= top + height):
            continue
        radius = max(0.0, min(float(shape.get("r", 0.0)), width / 2.0, height / 2.0))
        nearest_x = min(max(x, left + radius), left + width - radius)
        nearest_y = min(max(y, top + radius), top + height - radius)
        if (x - nearest_x) ** 2 + (y - nearest_y) ** 2 <= radius * radius + 1.0:
            return str(shape.get("id", ""))
    return ""


def caret_placement(caret, visible, width: float = PANEL_WIDTH, height: float = PANEL_HEIGHT):
    """Where the panel goes to show the capsule just under the caret.

    ``caret`` and ``visible`` are (x, y, w, h) in screen coordinates, origin at
    the bottom left. The capsule goes above the caret when there is not room
    for a card below it. Returns ``(panel_x, panel_y, anchor)``, the anchor in
    the page's coordinates, or ``None`` when neither side has room.
    """

    cx, cy, cw, ch = (float(v) for v in caret)
    vx, vy, vw, vh = (float(v) for v in visible)
    need = CARET_GAP + CARET_ROOM
    panel_x = min(max(cx + cw / 2.0 - width / 2.0, vx), vx + vw - width)
    anchor_x = round(cx + cw / 2.0 - panel_x, 1)
    if cy - vy >= need:
        panel_y = cy - CARET_GAP + CARET_PAD - height
        return panel_x, panel_y, {"mode": "top", "x": anchor_x, "y": CARET_PAD}
    if vy + vh - (cy + ch) >= need:
        panel_y = cy + ch + CARET_GAP - CARET_PAD
        return panel_x, panel_y, {"mode": "bottom", "x": anchor_x, "y": height - CARET_PAD}
    return None


class EditablePanel(NSPanel):
    """A non-activating panel that takes the keyboard only while a card is up.

    Non-activating keeps the app the text belongs to in front, so a paste still
    lands where the caret already is. The card's field needs real keystrokes
    and a working input method, so the panel may become key while it shows.
    """

    def canBecomeKeyWindow(self) -> bool:  # noqa: N802 - Cocoa selector
        return bool(getattr(self, "allow_key", False))


class FirstMouseWebView(WKWebView):
    """A web view that acts on the first click, though its panel is not key."""

    def acceptsFirstMouse_(self, event) -> bool:  # noqa: N802 - Cocoa selector
        del event
        return True


class FloatingOverlay(NSObject):
    """Owns the capsule's panel. Every method runs on the main thread."""

    def init(self):
        self = objc.super(FloatingOverlay, self).init()
        if self is None:
            return None
        self.panel = None
        self.page = None
        self.glass = None
        self.action_callback = None
        self.hover_callback = None
        self._ready = False
        self._state = "hidden"
        self._data: dict = {}
        self._rec_started = 0.0
        self._card = False
        self._handle = False
        self._anchor = {"mode": "bottom", "x": None, "y": None}
        self._caret = False
        self._back_to_bottom = False
        self._shapes: list[dict] = []
        self._shown = False
        self._over = ""
        self._pointer = None
        self._timer = None
        self._monitors: list = []
        self._followed_at = 0.0
        return self

    # ------------------------------------------------------------------ setup

    @objc.python_method
    def set_action_callback(self, callback: ActionCallback | None) -> None:
        """``callback(action, data)`` for buttons, card edits and the idle handle."""

        self.action_callback = callback

    @objc.python_method
    def set_hover_callback(self, callback: HoverCallback | None) -> None:
        """``callback(on)`` while the pointer is over the capsule (dismissal waits)."""

        self.hover_callback = callback

    @objc.python_method
    def setup(self) -> None:
        panel = EditablePanel.alloc().initWithContentRect_styleMask_backing_defer_(
            self._bottom_rect(self._pointer_screen()),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.allow_key = False
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(False)
        panel.setLevel_(NSStatusWindowLevel)
        panel.setIgnoresMouseEvents_(True)
        panel.setAcceptsMouseMovedEvents_(True)
        panel.setHidesOnDeactivate_(False)
        self.panel = panel
        self._set_spaces(capsule=False)

        root = FlippedView.alloc().initWithFrame_(NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT))
        panel.setContentView_(root)
        # Glass first, so it sits under the transparent page.
        self.glass = GlassLayer(root)
        self.page = WebPage(NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT), self._on_message, FirstMouseWebView)
        self.page.view.setAutoresizingMask_(18)
        root.addSubview_(self.page.view)
        self.page.load("overlay.html")

        NSWorkspace.sharedWorkspace().notificationCenter().addObserver_selector_name_object_(
            self, "accessibilityChanged:", _A11Y_CHANGED, None
        )
        panel.orderFrontRegardless()

    # ------------------------------------------------------------ public API

    @objc.python_method
    def show(self, state: str, **data) -> None:
        """Morph the capsule to ``state``; ``data`` fills in the state's details."""

        if state not in STATES:
            LOGGER.warning("Unknown capsule state %r", state)
            return
        was_hidden = self._state == "hidden"
        if state == "rec" and self._state != "rec":
            self._rec_started = time.monotonic() - float(data.get("elapsed") or 0.0)
        self._state = state
        self._data = dict(data)
        self._back_to_bottom = False
        if self.panel is None:
            return
        if was_hidden and not self._caret:
            self._follow_pointer(force=True)
        self._set_spaces(capsule=True)
        if state in CARD_STATES:
            self._begin_card()
        else:
            self._end_card()
            self.panel.orderFrontRegardless()
        self._send({"t": "show", "st": state, "d": data})
        if self._card:
            self._send({"t": "focus"})
        self._watch_pointer(True)

    @objc.python_method
    def hide(self) -> None:
        self._state = "hidden"
        self._data = {}
        self._end_card()
        self._send({"t": "hide"})
        if self._caret:
            if self._shown:
                # Back above the Dock once the capsule has faded out where it was.
                self._back_to_bottom = True
            else:
                self._anchor_bottom()

    @objc.python_method
    def end_edit(self) -> None:
        """Hand the keyboard back to the app it came from."""

        self._end_card()

    @objc.python_method
    def update_level(self, level: float) -> None:
        if self._state == "rec":
            self._send({"t": "level", "v": round(max(0.0, min(1.0, float(level))), 4)})

    @objc.python_method
    def set_handle(self, on: bool) -> None:
        """The thin line above the Dock that turns into a mic button on hover."""

        self._handle = bool(on)
        self._send_handle()
        self._watch_pointer(self._needs_pointer())

    @objc.python_method
    def set_anchor(self, mode: str, caret=None) -> None:
        """``"bottom"``: centred above the Dock. ``"caret"``: just under ``caret``,
        the caret's (x, y, w, h) in screen coordinates; bottom when it is unknown
        or there is no room around it."""

        self._back_to_bottom = False
        placement = None
        if mode == "caret" and caret is not None and self.panel is not None:
            screen = _screen_for_point(NSMakePoint(caret[0], caret[1]), NSScreen.screens()) or NSScreen.mainScreen()
            if screen is not None:
                frame = screen.visibleFrame()
                visible = (frame.origin.x, frame.origin.y, frame.size.width, frame.size.height)
                placement = caret_placement(caret, visible)
        if placement is None:
            self._anchor_bottom()
            return
        panel_x, panel_y, anchor = placement
        self._caret = True
        self._anchor = anchor
        self.panel.setFrame_display_(NSMakeRect(panel_x, panel_y, PANEL_WIDTH, PANEL_HEIGHT), True)
        self._send({"t": "anchor", **anchor})
        self._send_handle()

    @objc.python_method
    def is_card(self) -> bool:
        return self._card

    @objc.python_method
    def state(self) -> str:
        return self._state

    # --------------------------------------------------------------- bridge

    @objc.python_method
    def _on_message(self, message: dict) -> None:
        kind = message.get("t")
        if kind == "geo":
            self._on_geo(message.get("s") or [])
        elif kind == "act":
            data = {k: message[k] for k in ("text", "term", "kind", "on") if k in message}
            if "text" in data:
                data["text"] = str(data["text"] or "")
            self._emit(str(message.get("a") or ""), data)
        elif kind == "edit":
            # The card's text as it is being typed ("edit" is the bud that opens the card).
            self._emit("draft", {"text": str(message.get("text") or "")})
        elif kind == "field":
            self._emit("field", {"focus": bool(message.get("focus"))})
        elif kind == "hover":
            if self.hover_callback is not None:
                self.hover_callback(bool(message.get("on")))
        elif kind == "say":
            announce(str(message.get("text") or ""))
        elif kind == "ready":
            self._ready = True
            self._replay()

    @objc.python_method
    def _emit(self, action: str, data: dict) -> None:
        if action and self.action_callback is not None:
            self.action_callback(action, data)

    @objc.python_method
    def _send(self, message: dict) -> None:
        # Before the page is ready there is nobody to receive; _replay() catches up.
        if self.page is not None and self._ready:
            self.page.send(message)

    @objc.python_method
    def _replay(self) -> None:
        """Bring a freshly loaded page up to date (first load, or after a reload)."""

        self._push_env()
        self._send({"t": "anchor", **self._anchor})
        self._send_handle()
        if self._state != "hidden":
            data = dict(self._data)
            if self._state == "rec":
                data["elapsed"] = time.monotonic() - self._rec_started
            self._send({"t": "show", "st": self._state, "d": data})
            if self._card:
                self._send({"t": "focus"})

    @objc.python_method
    def _send_handle(self) -> None:
        self._send({"t": "handle", "on": self._handle and not self._caret})

    @objc.python_method
    def _push_env(self) -> None:
        env = accessibility_env()
        if self.glass is not None:
            self.glass.set_suppressed(env["rt"])
            if not env["rt"]:
                self.glass.apply(self._shapes)
        env["native"] = bool(self.glass is not None and self.glass.kind)
        env["lang"] = i18n.current()
        self._send({"t": "env", **env})

    @objc.python_method
    def relocalize(self) -> None:
        """The interface language changed; the page redraws what it shows in it."""

        self._push_env()

    def accessibilityChanged_(self, note) -> None:  # noqa: N802 - Cocoa selector
        del note
        self._push_env()

    # -------------------------------------------------------- shapes, pointer

    @objc.python_method
    def _on_geo(self, shapes: list[dict]) -> None:
        self._shapes = shapes
        if self.glass is not None:
            self.glass.apply(shapes)
        shown = any(s.get("id") not in ("handle", "hb") and float(s.get("a", 0)) > 0.01 for s in shapes)
        if self._shown and not shown:
            if self._back_to_bottom:
                self._back_to_bottom = False
                self._anchor_bottom()
            if self._state == "hidden":
                # The idle handle stays out of full-screen apps.
                self._set_spaces(capsule=False)
        self._shown = shown
        self._watch_pointer(self._needs_pointer())
        self._refresh_pointer()

    @objc.python_method
    def _needs_pointer(self) -> bool:
        return self._shown or self._state != "hidden" or (self._handle and not self._caret)

    @objc.python_method
    def _watch_pointer(self, on: bool) -> None:
        """Follow the pointer only while something can be hovered or clicked.

        Mouse moves elsewhere come from a global monitor, so an idle handle
        costs nothing while the mouse is still; a timer runs only while the
        pointer is over the capsule, where the panel itself gets the events.
        """

        if on and not self._monitors and self._timer is None:
            try:
                monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                    _MOUSE_MOVED_MASK, lambda event: self._on_mouse_moved()
                )
            except Exception:
                monitor = None
            if monitor is not None:
                self._monitors.append(monitor)
            else:
                self._set_timer(True)
        elif not on and self._monitors:
            for monitor in self._monitors:
                NSEvent.removeMonitor_(monitor)
            self._monitors = []
        if not on:
            self._set_timer(False)
            self._set_over("")
            if self._pointer is not None:
                self._pointer = None
                self._send({"t": "pointer", "x": None, "y": None})

    @objc.python_method
    def _set_timer(self, on: bool) -> None:
        if on and self._timer is None:
            timer = NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
                POINTER_INTERVAL, self, "pollPointer:", None, True
            )
            NSRunLoop.mainRunLoop().addTimer_forMode_(timer, NSRunLoopCommonModes)
            self._timer = timer
        elif not on and self._timer is not None:
            self._timer.invalidate()
            self._timer = None

    def pollPointer_(self, timer) -> None:  # noqa: N802 - Cocoa selector
        del timer
        self._on_mouse_moved()

    @objc.python_method
    def _on_mouse_moved(self) -> None:
        if not self._caret and not self._card:
            self._follow_pointer()
        self._refresh_pointer()

    @objc.python_method
    def _refresh_pointer(self) -> None:
        if self.panel is None:
            return
        point = NSEvent.mouseLocation()
        frame = self.panel.frame()
        x = float(point.x - frame.origin.x)
        y = float(frame.size.height - (point.y - frame.origin.y))
        inside = 0.0 <= x <= frame.size.width and 0.0 <= y <= frame.size.height
        self._set_over(hit_shape(self._shapes, x, y) if inside else "")
        position = (round(x, 1), round(y, 1)) if inside else None
        if position != self._pointer:
            self._pointer = position
            self._send({"t": "pointer", "x": position[0] if position else None, "y": position[1] if position else None})

    @objc.python_method
    def _set_over(self, shape_id: str) -> None:
        over = bool(shape_id)
        if over == bool(self._over):
            self._over = shape_id
            return
        # A drag that started on the capsule keeps its mouse events until release.
        if not over and NSEvent.pressedMouseButtons():
            return
        self._over = shape_id
        if self.panel is not None:
            self.panel.setIgnoresMouseEvents_(not over)
        # Over the capsule the panel gets the mouse events itself, and the global
        # monitor goes quiet, so hover is polled until the pointer leaves.
        self._set_timer(over or not self._monitors and self._needs_pointer())

    # ------------------------------------------------------------ placement

    @objc.python_method
    def _pointer_screen(self):
        return _screen_for_point(NSEvent.mouseLocation(), NSScreen.screens()) or NSScreen.mainScreen()

    @objc.python_method
    def _bottom_rect(self, screen):
        if screen is None:
            return NSMakeRect(0, 0, PANEL_WIDTH, PANEL_HEIGHT)
        frame = screen.visibleFrame()
        return NSMakeRect(
            frame.origin.x + (frame.size.width - PANEL_WIDTH) / 2.0,
            frame.origin.y,
            PANEL_WIDTH,
            PANEL_HEIGHT,
        )

    @objc.python_method
    def _follow_pointer(self, force: bool = False) -> None:
        """Keep the panel on the display the pointer is on."""

        now = time.monotonic()
        if self.panel is None or (not force and now - self._followed_at < FOLLOW_INTERVAL):
            return
        self._followed_at = now
        rect = self._bottom_rect(self._pointer_screen())
        current = self.panel.frame()
        if abs(current.origin.x - rect.origin.x) >= 1.0 or abs(current.origin.y - rect.origin.y) >= 1.0:
            self.panel.setFrame_display_(rect, True)

    @objc.python_method
    def _anchor_bottom(self) -> None:
        self._caret = False
        self._anchor = {"mode": "bottom", "x": None, "y": None}
        self._follow_pointer(force=True)
        self._send({"t": "anchor", **self._anchor})
        self._send_handle()

    @objc.python_method
    def _set_spaces(self, capsule: bool) -> None:
        behavior = NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorIgnoresCycle
        if capsule:
            behavior |= NSWindowCollectionBehaviorFullScreenAuxiliary
        if self.panel is not None:
            self.panel.setCollectionBehavior_(behavior)

    # ----------------------------------------------------------------- cards

    @objc.python_method
    def _begin_card(self) -> None:
        self._card = True
        self.panel.allow_key = True
        # An input method draws its candidate list at window level 20, under the
        # status level (25) the capsule normally sits at; drop below it while
        # something can be typed.
        self.panel.setLevel_(NSFloatingWindowLevel)
        self.panel.makeKeyAndOrderFront_(None)
        if self.page is not None:
            self.panel.makeFirstResponder_(self.page.view)

    @objc.python_method
    def _end_card(self) -> None:
        if not self._card:
            return
        self._card = False
        if self.panel is None:
            return
        self.panel.allow_key = False
        self.panel.setLevel_(NSStatusWindowLevel)
        # Ordering out and back in drops key status without disturbing what is
        # in front; a panel left key would go on swallowing keystrokes.
        self.panel.orderOut_(None)
        self.panel.orderFrontRegardless()
