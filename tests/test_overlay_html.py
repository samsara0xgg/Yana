from __future__ import annotations

import re
from types import SimpleNamespace

from typeless_local.overlay import (
    ACTIONS,
    CARET_GAP,
    PANEL_HEIGHT,
    PANEL_WIDTH,
    STATES,
    FloatingOverlay,
    _screen_for_point,
    caret_placement,
    hit_shape,
)
from typeless_local.webview import dispatch, web_root


def _read(name: str) -> str:
    return (web_root() / name).read_text(encoding="utf-8")


def test_capsule_page_loads_its_assets() -> None:
    html = _read("overlay.html")
    for name in ("kit.css", "overlay.css", "kit.js", "overlay.js"):
        assert name in html
        assert (web_root() / name).is_file()


def test_capsule_page_draws_every_state_the_app_can_show() -> None:
    drawn = set(re.findall(r"case '([a-z-]+)':", _read("overlay.js")))
    assert STATES <= drawn


def test_capsule_page_posts_only_actions_the_app_knows() -> None:
    page = _read("overlay.js") + _read("overlay.html")
    posted = set(re.findall(r'data-act="([a-z]+)"', page)) | set(re.findall(r"\['(done|close|replace)'", page))
    assert posted == ACTIONS


def test_pages_post_json_strings_the_bridge_parses() -> None:
    assert "postMessage(JSON.stringify(msg))" in _read("kit.js")


def test_hit_shape_follows_the_rounded_corners() -> None:
    pill = {"id": "cap", "x": 100, "y": 50, "w": 200, "h": 40, "r": 20, "hit": True}
    assert hit_shape([pill], 200, 70) == "cap"
    assert hit_shape([pill], 120, 51) == "cap"
    # Inside the bounding box but outside the rounded corner: the app below gets the click.
    assert hit_shape([pill], 101, 51) == ""
    assert hit_shape([pill], 99, 70) == ""
    assert hit_shape([dict(pill, hit=False)], 200, 70) == ""


def test_hit_shape_prefers_the_shape_drawn_last() -> None:
    cap = {"id": "cap", "x": 0, "y": 0, "w": 100, "h": 40, "r": 0, "hit": True}
    bud = {"id": "b1", "x": 90, "y": 2, "w": 80, "h": 36, "r": 0, "hit": True}
    assert hit_shape([cap, bud], 95, 20) == "b1"


VISIBLE = (0, 0, 1440, 875)


def test_caret_placement_puts_the_capsule_just_under_the_caret() -> None:
    panel_x, panel_y, anchor = caret_placement((400, 600, 2, 18), VISIBLE)
    assert anchor["mode"] == "top"
    assert panel_y + PANEL_HEIGHT - anchor["y"] == 600 - CARET_GAP
    assert panel_x + anchor["x"] == 401


def test_caret_placement_goes_above_a_caret_near_the_bottom() -> None:
    _, panel_y, anchor = caret_placement((400, 100, 2, 18), VISIBLE)
    assert anchor["mode"] == "bottom"
    assert panel_y + PANEL_HEIGHT - anchor["y"] == 100 + 18 + CARET_GAP


def test_caret_placement_keeps_the_panel_on_the_screen() -> None:
    panel_x, _, anchor = caret_placement((5, 600, 2, 18), VISIBLE)
    assert panel_x == 0 and anchor["x"] == 6
    panel_x, _, anchor = caret_placement((1435, 600, 2, 18), VISIBLE)
    assert panel_x == 1440 - PANEL_WIDTH and panel_x + anchor["x"] == 1436


def test_caret_placement_gives_up_when_neither_side_has_room() -> None:
    assert caret_placement((10, 150, 2, 18), (0, 0, 800, 400)) is None


def _overlay():
    overlay = FloatingOverlay.alloc().init()
    sent: list[dict] = []
    overlay.page = SimpleNamespace(send=sent.append)
    overlay._ready = True
    return overlay, sent


def test_bridge_messages_reach_the_app() -> None:
    overlay, _ = _overlay()
    actions, hovers = [], []
    overlay.set_action_callback(lambda action, data: actions.append((action, data)))
    overlay.set_hover_callback(hovers.append)

    for body in (
        '{"t":"act","a":"undo"}',
        '{"t":"act","a":"replace","text":"改好的"}',
        '{"t":"edit","text":"改"}',
        '{"t":"hover","on":true}',
        "not json",
        '{"t":"jserror","msg":"boom"}',
        '["not", "a", "message"]',
    ):
        dispatch(body, overlay._on_message)

    assert actions == [("undo", {}), ("replace", {"text": "改好的"}), ("draft", {"text": "改"})]
    assert hovers == [True]


def test_a_reloaded_page_is_brought_up_to_date() -> None:
    overlay, sent = _overlay()
    overlay._ready = False
    overlay.show("inserted", n=12)
    assert sent == []

    overlay._on_message({"t": "ready"})

    assert [m["t"] for m in sent] == ["env", "anchor", "handle", "show"]
    assert sent[-1] == {"t": "show", "st": "inserted", "d": {"n": 12}}


def test_replayed_recording_keeps_its_elapsed_time() -> None:
    overlay, sent = _overlay()
    overlay.show("rec", mode="click", max=540)
    overlay._rec_started -= 5
    sent.clear()

    overlay._on_message({"t": "ready"})

    assert sent[-1]["st"] == "rec"
    assert 4.9 < sent[-1]["d"]["elapsed"] < 6


def test_levels_are_only_sent_while_recording() -> None:
    overlay, sent = _overlay()
    overlay.update_level(0.5)
    overlay.show("rec", mode="hold")
    overlay.update_level(3.0)
    assert [m for m in sent if m["t"] == "level"] == [{"t": "level", "v": 1.0}]


def test_unknown_states_are_not_sent() -> None:
    overlay, sent = _overlay()
    overlay.show("thinking")
    assert sent == [] and overlay.state() == "hidden"


def test_capsule_returns_above_the_dock_after_fading_out_at_the_caret() -> None:
    overlay, sent = _overlay()
    overlay._caret = True
    overlay._anchor = {"mode": "top", "x": 100, "y": 16}
    cap = {"id": "cap", "x": 80, "y": 16, "w": 40, "h": 40, "r": 20, "a": 1, "glass": True, "hit": True}
    overlay._on_geo([cap])
    overlay.hide()
    overlay._on_geo([dict(cap, a=0.4)])
    assert overlay._caret

    overlay._on_geo([])

    assert not overlay._caret
    assert {"t": "anchor", "mode": "bottom", "x": None, "y": None} in sent
    assert sent[-1] == {"t": "handle", "on": False}


def test_handle_is_hidden_while_the_capsule_follows_the_caret() -> None:
    overlay, sent = _overlay()
    overlay.set_handle(True)
    overlay._caret = True
    overlay._send_handle()
    assert [m["on"] for m in sent if m["t"] == "handle"] == [True, False]


def test_screen_for_point_picks_the_display_under_the_pointer() -> None:
    """The overlay must follow the pointer across displays.

    ``FloatingOverlay.setup`` frames the panel once from ``NSScreen.mainScreen()``,
    so before this the overlay stayed on whichever display was main when the app
    launched. Geometry below is Allen's real 2026-09-12 layout: the BenQ at the
    origin and the built-in display to its left at a negative x.
    """

    def screen(x, y, w, h):
        rect = SimpleNamespace(origin=SimpleNamespace(x=x, y=y), size=SimpleNamespace(width=w, height=h))
        return SimpleNamespace(frame=lambda rect=rect: rect, tag=(x, y))

    benq = screen(0, 0, 1920, 1080)
    built_in = screen(-1352, 0, 1352, 878)
    screens = [benq, built_in]

    assert _screen_for_point(SimpleNamespace(x=952, y=272), screens) is benq
    assert _screen_for_point(SimpleNamespace(x=-600, y=400), screens) is built_in
    # The menu bar sits above visibleFrame but still belongs to that display.
    assert _screen_for_point(SimpleNamespace(x=100, y=1070), screens) is benq
    # Exactly on the shared edge belongs to the display that owns that origin.
    assert _screen_for_point(SimpleNamespace(x=0, y=500), screens) is benq
    assert _screen_for_point(SimpleNamespace(x=-1, y=500), screens) is built_in
    # Off every display.
    assert _screen_for_point(SimpleNamespace(x=5000, y=5000), screens) is None
