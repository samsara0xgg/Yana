"""The app's windows: Settings, History and the first-run guide.

Each window is an ordinary titled NSWindow whose whole content is a web page
(web/settings.html, history.html, onboarding.html), drawn with the same kit as
the capsule. The pages only draw and report; everything they show is built
here from the app, and everything they ask for is done here through the app,
so this module is where the windows' behaviour lives and is tested.

AppKit is only touched by ``WebWindow`` and ``install_main_menu``; the rest is
plain Python run on the main thread (bridge messages arrive there), with
anything slow (the keychain, a network test, polling a permission) moved to a
thread whose result comes back through ``app._call_ui``.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable
from urllib.parse import urlparse

from typeless_local import app_version, brand, cloud_asr, devices, history, i18n, keychain, login_item, permissions, preferences, reach, trial, usage, vocab
from typeless_local.asr import mlx_whisper_repo
from typeless_local.config import preset_names, refine_config_for
from typeless_local.mac_integration import FocusContext, has_accessibility_trust, request_accessibility_trust, set_clipboard_text
from typeless_local.i18n import t
from typeless_local.trace import load_corrections

LOGGER = logging.getLogger(__name__)

SETTINGS_PANES = ("general", "dictation", "keys", "model", "vocab", "audio", "usage", "privacy", "about")


def _open_urls() -> dict[str, str]:
    """The only web addresses a page may ask to open, by name."""

    urls = {"releases": brand.RELEASES_URL, "repo": brand.REPO_URL, "github": brand.AUTHOR_GITHUB}
    urls.update({f"link{i}": url for i, (_, url) in enumerate(brand.AUTHOR_LINKS)})
    return {name: url for name, url in urls.items() if url.startswith("https://")}


SUGGESTIONS = 8
HISTORY_ROWS = 500
FIXES = 6
POLL_S = 1.0
POLL_TIMEOUT_S = 300.0
METER_EVERY_S = 0.05  # the guide's level meter: twenty updates a second is smooth enough
_SERVICES = {
    "openai": "OpenAI",
    "deepseek": "DeepSeek",
    "anthropic": "Anthropic",
    "googleapis": "Google",
    "moonshot": "Moonshot",
    "dashscope": ("通义千问", "Qwen"),
    "bigmodel": ("智谱", "Zhipu"),
}


def service_name(base_url: str | None) -> str:
    """"OpenAI" for https://api.openai.com/v1, and so on; the host otherwise."""

    host = urlparse(base_url or "https://api.openai.com").hostname or ""
    for key, name in _SERVICES.items():
        if key in host:
            return t(*name) if isinstance(name, tuple) else name
    return host or "OpenAI"


def env_file_keys(path: Path | None) -> set[str]:
    """Names that still have a value in the plain-text env file."""

    if path is None or not Path(path).exists():
        return set()
    names = set()
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#") and value.strip():
                names.add(key.strip())
    except OSError:
        pass
    return names


def prices(jarvis: dict) -> dict:
    """``llm.prices`` from the engine config: per-model price overrides."""

    found = ((jarvis or {}).get("llm") or {}).get("prices")
    return found if isinstance(found, dict) else {}


def short_model(repo: str) -> str:
    return (repo or "").rstrip("/").split("/")[-1]


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-'+#]*|\s+|[^\sA-Za-z0-9]")
_EDGE = " \t\n,.;:!?，。；：！？、“”\"'"


def fix_terms(before: str, after: str) -> list[tuple[str, str]]:
    """Words a hand edit put in, as (was, now): whole English words, never lone characters.

    The corrections log keeps character-level fragments ("o" -> "PyO"), which
    say what changed but not which word to learn.
    """

    old, new = _TOKEN.findall(before or ""), _TOKEN.findall(after or "")
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal" or j1 == j2:
            continue
        wrong, right = "".join(old[i1:i2]).strip(_EDGE), "".join(new[j1:j2]).strip(_EDGE)
        if len(right) < 2 or wrong.lower() == right.lower():
            continue
        out.append((wrong, right))
    return out


_CJK = re.compile(r"[\u4e00-\u9fff]")


def learned_fixes(pasted: str, sent: str) -> list[tuple[str, str]]:
    """Mishears the user fixed by hand before sending, as (was, now); [] unless every change is a clean swap."""

    def tokens(text: str) -> list[str]:
        return [" " if tok.isspace() else tok for tok in _TOKEN.findall(text or "")]

    old, new = tokens(pasted), tokens(sent)
    ops = [op for op in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if op[0] != "equal"]
    # The field can hold more than the paste: text typed or dictated before or after it.
    ops = [op for op in ops if not (op[0] == "insert" and op[1] in (0, len(old)))]
    if len(ops) > 2 or any(op[0] != "replace" for op in ops):
        return []
    out = []
    for _, i1, i2, j1, j2 in ops:
        wrong, right = "".join(old[i1:i2]).strip(_EDGE), "".join(new[j1:j2]).strip(_EDGE)
        if len(right) == 1 and _CJK.fullmatch(right):
            # ponytail: no word segmentation; borrow the CJK neighbours so a one-character fix reads as a word.
            if i1 and _CJK.fullmatch(old[i1 - 1]):
                wrong, right = old[i1 - 1] + wrong, old[i1 - 1] + right
            if i2 < len(old) and _CJK.fullmatch(old[i2]):
                wrong, right = wrong + old[i2], right + old[i2]
        if (
            not 2 <= len(right) <= 24
            or not wrong
            or wrong.lower() == right.lower()
            or not re.search(r"[^\W\d_]", right)
            or sum(not x.isspace() for x in _TOKEN.findall(wrong)) > 4
            or sum(not x.isspace() for x in _TOKEN.findall(right)) > 4
            or (wrong.isascii() and not re.search(r"[A-Za-z]", right))
        ):
            continue
        out.append((wrong, right))
    return out


class Windows:
    """Opens the windows and answers their pages."""

    def __init__(self, app, factory: Callable | None = None) -> None:
        self.app = app
        self._factory = factory or WebWindow
        self.settings = None
        self.history = None
        self.onboarding = None
        self._f5_conflict: bool | None = None
        self._polls: set[str] = set()
        self._meter = None  # the guide's microphone level stream, open only while its page asks

    @staticmethod
    def _titles() -> dict[str, str]:
        return {
            "settings": t("设置", "Settings"),
            "history": t("历史记录", "History"),
            "onboarding": brand.join(t("欢迎使用", "Welcome to"), brand.display_name()),
        }

    # ---------------------------------------------------------- opening

    def _window(self, name: str, **kwargs):
        window = getattr(self, name)
        if window is None:
            window = self._factory(**kwargs)
            setattr(self, name, window)
        return window

    def show_settings(self, pane: str | None = None) -> None:
        reopening = self._hidden("settings")
        window = self._window(
            "settings", page="settings.html", title=self._titles()["settings"], size=(780, 580), min_size=(600, 420),
            toolbar=True, on_message=self._settings_message,
        )
        window.show()
        if reopening:
            self._f5_conflict = None
            window.send(self.settings_state())
        if pane in SETTINGS_PANES:
            window.send({"t": "pane", "id": pane})

    def show_history(self) -> None:
        reopening = self._hidden("history")
        window = self._window(
            "history", page="history.html", title=self._titles()["history"], size=(860, 580), min_size=(620, 400),
            toolbar=True, on_message=self._history_message,
        )
        window.show()
        if reopening:
            window.send({"t": "items", **self.history_payload()})

    def show_onboarding(self) -> None:
        reopening = self._hidden("onboarding")
        window = self._window(
            "onboarding", page="onboarding.html", title=self._titles()["onboarding"], size=(620, 560),
            resizable=False, on_message=self._onboarding_message,
        )
        window.show()
        if reopening:
            window.send(self.onboarding_state())

    def _hidden(self, name: str) -> bool:
        """Built and closed since: the page stayed loaded but missed every refresh."""

        window = getattr(self, name)
        return window is not None and not window.visible()

    def _visible(self, name: str) -> bool:
        window = getattr(self, name)
        return window is not None and window.visible()

    def refresh(self, history: bool = False) -> None:
        """Something a window shows changed elsewhere (the menu, a dictation)."""

        if self._visible("settings"):
            self.settings.send(self.settings_state())
        if history and self._visible("history"):
            self.history.send({"t": "items", **self.history_payload()})

    def relocalize(self) -> None:
        """The interface language changed: every page built so far redraws in it.

        Hidden pages too, since reopening one only resends its state, not its env.
        """

        for name, title in self._titles().items():
            window = getattr(self, name)
            if window is None:
                continue
            window.set_title(title)
            self._env(window)
        if self.settings is not None:
            self.settings.send(self.settings_state())
        if self.history is not None:
            self.history.send({"t": "items", **self.history_payload()})
        if self.onboarding is not None:
            self.onboarding.send(self.onboarding_state())

    def download(self, fraction: float, eta: str = "", done: bool = False, error: bool = False, why: str = "") -> None:
        if self._visible("onboarding"):
            self.onboarding.send({"t": "download", "p": round(fraction, 3), "eta": eta, "done": done, "error": error, "why": why})

    def _later(self, window, message: dict) -> None:
        """Send from a worker thread."""

        self.app._call_ui(window.send, message)

    def _run(self, job: Callable, name: str) -> None:
        threading.Thread(target=job, daemon=True, name=name).start()

    # ------------------------------------------------------------- env

    def _env(self, window) -> None:
        from typeless_local.webview import accessibility_env  # noqa: PLC0415

        env = accessibility_env()
        # Reduce Transparency: the page draws solid fills and the glass stays hidden.
        window.suppress_glass(bool(env.get("rt")))
        env["native"] = bool(getattr(window, "has_glass", False)) and not env.get("rt")
        env["lang"] = i18n.current()
        window.send({"t": "env", **env})

    # ---------------------------------------------------------- settings

    def settings_state(self) -> dict:
        app = self.app
        config = app.config
        prefs = app.prefs
        user_paths = getattr(config, "user_paths", None)
        jarvis = getattr(config, "jarvis_config", {}) or {}
        db = user_paths.trace_db_path if user_paths is not None else None
        in_file = env_file_keys(user_paths.env_path if user_paths is not None else None)

        presets, keys, seen = [], [], set()
        for name in preset_names(jarvis):
            try:
                refine = refine_config_for(jarvis, name)
            except Exception:
                continue
            is_trial = name == trial.PRESET
            service = t("免费试用", "Free trial") if is_trial else service_name(refine.base_url)
            median = history.median_refine_ms(db, refine.model) if db is not None and prefs.save_history else None
            presets.append(
                {
                    "name": name,
                    "model": refine.model,
                    "service": service,
                    "env": refine.api_key_env,
                    "hasKey": bool(os.environ.get(refine.api_key_env or "")),
                    "median": median,
                }
            )
            env = refine.api_key_env
            if env and env not in seen and not is_trial:
                seen.add(env)
                value = os.environ.get(env, "")
                where = "env" if env in in_file else ("keychain" if value else "")
                keys.append({"env": env, "service": service, "hint": keychain.masked(value), "where": where})

        # Only the key the one model in use reads (on the trial, the user's own OpenAI key).
        active = getattr(config.refine, "preset", "")
        wanted = trial.OWN_KEY_ENV if active == trial.PRESET else getattr(config.refine, "api_key_env", "")
        keys = [key for key in keys if key["env"] == wanted]
        cloud_key = os.environ.get(cloud_asr.KEY_ENV, "")
        asr_key = {
            "env": cloud_asr.KEY_ENV,
            "service": t("阿里云百炼", "Alibaba Cloud Model Studio"),
            "hint": keychain.masked(cloud_key),
            "where": "env" if cloud_asr.KEY_ENV in in_file else ("keychain" if cloud_key else ""),
        }

        asr_config = jarvis.get("asr") or {}
        ducker = getattr(app, "audio_ducker", None)
        from typeless_local import devices  # noqa: PLC0415

        if self._f5_conflict is None:
            self._f5_conflict = permissions.system_dictation_uses_f5()
        return {
            "t": "state",
            "name": brand.display_name(),
            "version": app_version(),
            "prefs": prefs.to_dict(),
            "choices": {key: list(values) for key, values in preferences.CHOICES.items()},
            "login": login_item.status(),
            "language": str(asr_config.get("language") or ""),
            "asrModel": short_model(mlx_whisper_repo(asr_config)) or str(getattr(app.asr, "model_name", "") or ""),
            "duck": bool(getattr(ducker, "enabled", True)),
            "duckUnsupported": bool(getattr(ducker, "unsupported", False)),
            "presets": presets,
            "active": getattr(config.refine, "preset", ""),
            "keys": keys,
            "asrKey": asr_key,
            "trial": {"on": getattr(config.refine, "preset", "") == trial.PRESET, "over": bool(getattr(app, "_trial_over", False))},
            "inputs": devices.list_input_devices(),
            "input": getattr(config, "input_device", "") or "",
            "vocab": self._vocab_state(),
            "history": {"count": history.count_sessions(db) if db is not None else 0},
            "usage": usage.summary(db, prices(jarvis)) if db is not None and prefs.save_history else None,
            "f5": bool(self._f5_conflict),
            "author": {"name": brand.AUTHOR_NAME, "github": brand.AUTHOR_GITHUB, "links": [list(link) for link in brand.AUTHOR_LINKS]},
        }

    def _vocab_state(self) -> dict:
        user_paths = getattr(self.app.config, "user_paths", None)
        if user_paths is None:
            return {"mine": [], "suggest": [], "fixes": [], "learned": []}
        sections = vocab._load_sections(Path(user_paths.vocab_path))
        mine = sections["user"]
        known = {term.lower() for term in mine + sections["rejected"]}
        candidates = [term for term in sections["auto"] if term.lower() not in known][:SUGGESTIONS]
        counts = history.term_counts(user_paths.trace_db_path, candidates) if candidates and self.app.prefs.save_history else {}
        suggest = [{"term": term, "n": counts.get(term, 0)} for term in candidates]
        fixes, rights = [], set()
        corrections = load_corrections(user_paths.corrections_path) if user_paths.corrections_path.exists() else []
        for entry in reversed(corrections):
            for wrong, right in fix_terms(str(entry.get("before") or ""), str(entry.get("after") or "")):
                if len(right) > 24 or right in rights:
                    continue
                rights.add(right)
                fixes.append({"wrong": wrong, "right": right, "at": str(entry.get("at") or "")[5:16]})
                if len(fixes) >= FIXES:
                    break
            if len(fixes) >= FIXES:
                break
        learned = [
            {"wrong": e["was"], "right": e["term"], "n": e["n"], "at": e["at"][5:16].replace("T", " ")}
            for e in reversed(sections["learned"]) if e["term"] in mine
        ]
        return {"mine": mine, "suggest": suggest, "fixes": fixes, "learned": learned}

    def _settings_message(self, msg: dict) -> None:
        kind = msg.get("t")
        window = self.settings
        if kind == "ready":
            self._f5_conflict = None
            self._env(window)
            window.send(self.settings_state())
        elif kind == "geo":
            window.apply_glass(msg.get("s") or [])
        elif kind == "set":
            self._apply_setting(str(msg.get("key") or ""), msg.get("value"))
            window.send(self.settings_state())
        elif kind == "key":
            self._save_key(window, str(msg.get("env") or ""), str(msg.get("value") or ""), test=False)
        elif kind == "migrate":
            self._migrate_keys(window)
        elif kind == "test":
            self._test_connection(window)
        elif kind == "vocab":
            self._edit_vocab(window, str(msg.get("op") or ""), str(msg.get("term") or ""))
        elif kind == "open":
            self._open(str(msg.get("what") or ""))
        elif kind == "feedback":
            self._send_feedback(window, msg)
        elif kind == "count":
            days = int(msg.get("days") or 0)
            db = self._db()
            count = history.count_sessions(db, older_than_days=days) if db is not None and days > 0 else 0
            window.send({"t": "count", "days": days, "n": count})
        elif kind == "clear":
            db = self._db()
            if db is not None:
                removed = history.clear_history(db)
                LOGGER.info("Cleared %d dictations from the history", removed)
            self.refresh(history=True)

    def _send_feedback(self, window, msg: dict) -> None:
        message = str(msg.get("message") or "").strip()
        if not message:
            return
        self.app.send_feedback(
            message, str(msg.get("email") or ""), bool(msg.get("diagnostics")),
            lambda ok, error: self._later(window, {"t": "feedbackResult", "ok": ok, "msg": error}),
        )

    def _db(self) -> Path | None:
        user_paths = getattr(self.app.config, "user_paths", None)
        return user_paths.trace_db_path if user_paths is not None else None

    def _apply_setting(self, key: str, value) -> None:
        app = self.app
        try:
            if key == "login":
                result = login_item.set_enabled(bool(value))
                if result == "requires_approval":
                    login_item.open_system_settings()
            elif key == "language":
                app.set_language(str(value or ""))
            elif key == "duck":
                app.set_ducking(bool(value))
            elif key == "preset":
                app.select_model(str(value))
            elif key == "input":
                app.select_input_device(str(value or ""))
            elif key in ("f5_hotkey", "right_command_hotkey") and not value:
                # Never leave the app with no shortcut at all.
                other = "right_command_hotkey" if key == "f5_hotkey" else "f5_hotkey"
                if getattr(app.prefs, other):
                    app.set_preference(key, False)
                else:
                    self.refresh()
            else:
                app.set_preference(key, value)
        except ValueError:
            LOGGER.warning("Settings sent an invalid value: %s=%r", key, value)
        except Exception:
            LOGGER.exception("Could not apply %s", key)

    def _save_key(self, window, env: str, value: str, *, test: bool) -> None:
        value = value.strip()
        if not env or not keychain.is_valid_key(value):
            msg = t("这不像一个 API Key：应该是一整串没有空格的字符。", "That doesn't look like an API key: it should be one string with no spaces.")
            window.send({"t": "keyResult", "env": env, "ok": False, "msg": msg})
            return

        def job() -> None:
            result = {"t": "keyResult", "env": env, "ok": True, "msg": t("已保存到钥匙串", "Saved to the keychain")}
            try:
                self.app.store_api_key(env, value)
                if test:
                    ok, ms, error = self.test_preset(self._preset_for(env))
                    connected = t(f"已连接 · 往返 {ms / 1000:.1f} 秒", f"Connected · {ms / 1000:.1f} s round trip")
                    result.update(ok=ok, ms=ms, msg=connected if ok else error)
            except Exception:
                LOGGER.exception("Saving %s failed", env)
                result.update(ok=False, msg=t("没能保存，详情见日志。", "Couldn't save it; the log has details."))
            self._later(window, result)
            if window is self.settings:
                self.app._call_ui(self.refresh)
            elif self._visible("onboarding"):
                self._later(window, self.onboarding_state())

        self._run(job, "save-key")

    def _preset_for(self, env: str) -> str:
        config = self.app.config
        if getattr(config.refine, "api_key_env", "") == env:
            return config.refine.preset
        jarvis = getattr(config, "jarvis_config", {}) or {}
        for name in preset_names(jarvis):
            try:
                if refine_config_for(jarvis, name).api_key_env == env:
                    return name
            except Exception:
                continue
        return config.refine.preset

    def _migrate_keys(self, window) -> None:
        user_paths = getattr(self.app.config, "user_paths", None)
        if user_paths is None:
            return

        def job() -> None:
            from typeless_local.app import api_key_names  # noqa: PLC0415

            moved = keychain.migrate_env_file(Path(user_paths.env_path), api_key_names(self.app.config))
            LOGGER.info("Moved %d key(s) to the keychain", len(moved))
            self.app._call_ui(self.refresh)

        self._run(job, "migrate-keys")

    def test_preset(self, preset: str) -> tuple[bool, int, str]:
        """One small real request through ``preset``: (worked, round trip ms, why not)."""

        from typeless_local.app import _failure_kind  # noqa: PLC0415
        from typeless_local.refine import MissingAPIKey, TextRefiner, TrialUnavailable  # noqa: PLC0415

        jarvis = getattr(self.app.config, "jarvis_config", {}) or {}
        try:
            refiner = TextRefiner(refine_config_for(jarvis, preset))
            started = time.monotonic()
            result = refiner.refine("嗯，测试一下连接", FocusContext(app_name="", window_title="", selected_text=""))
            ms = int((time.monotonic() - started) * 1000)
        except TrialUnavailable as exc:
            why = {
                "trial_region": t("免费试用只在美国和加拿大提供。", "The free trial is only offered in the US and Canada."),
                "trial_paused": t("免费试用本月已暂停。", "The free trial is paused this month."),
                "trial_ip": t("这个网络今天的免费额度用完了，明天再试。", "This network's free trial is done for today; try again tomorrow."),
            }
            return False, 0, why.get(exc.code, t("免费试用已经用完了。", "The free trial is used up."))
        except MissingAPIKey:
            return False, 0, t("还没有填 API Key。", "There is no API key yet.")
        except Exception as exc:
            LOGGER.warning("Connection test for %s failed", preset, exc_info=True)
            kind = _failure_kind(exc)
            if kind == "credit":
                return False, 0, t("OpenAI 账户余额用完了，充值后再试。", "Your OpenAI account is out of credit. Add credit and try again.")
            text = str(exc)
            if kind == "badkey" or "401" in text:
                return False, 0, t("API Key 不对，服务拒绝了请求。", "The service rejected the API key.")
            if kind == "timeout":
                return False, 0, t("请求超时，检查一下网络。", "The request timed out. Check the network.")
            # The raw error can name the model; say only what kind of failure it was.
            return False, 0, t("连不上服务，检查一下网络后再试。", "Can't reach the service. Check the network and try again.") + f" ({type(exc).__name__})"
        if getattr(result, "fallback", ""):
            return False, ms, t("连上了，但模型没有正常返回。", "Connected, but the model gave no proper answer.")
        return True, ms, ""

    def _test_connection(self, window) -> None:
        preset = getattr(self.app.config.refine, "preset", "")

        def job() -> None:
            ok, ms, error = self.test_preset(preset)
            self._later(window, {"t": "testResult", "ok": ok, "ms": ms, "msg": error, "preset": preset})
            self.app._call_ui(self.refresh)

        self._run(job, "test-connection")

    def _edit_vocab(self, window, op: str, term: str) -> None:
        term = term.strip()
        user_paths = getattr(self.app.config, "user_paths", None)
        if not term or user_paths is None:
            return
        path = Path(user_paths.vocab_path)
        mine = vocab.load_user_terms(path)
        message = ""
        if op == "add":
            if term in mine:
                message = t(f"“{term}”已经在词库里了。", f"“{term}” is already in the vocabulary.")
            else:
                vocab.save_user_terms(path, mine + [term])
        elif op == "remove":
            vocab.save_user_terms(path, [word for word in mine if word != term])
        elif op == "reject":
            vocab.reject_term(path, term)
        elif op == "unlearn":
            vocab.unlearn_term(path, term)
        else:
            return
        self.app.reload_vocab()
        if window is not None:
            window.send({"t": "vocabResult", "msg": message})
        self.refresh()

    def _open(self, what: str) -> None:
        user_paths = getattr(self.app.config, "user_paths", None)
        if what == "data" and user_paths is not None:
            subprocess.Popen(["/usr/bin/open", str(user_paths.config_dir)])
        elif what == "log":
            self.app.show_log()
        elif what == "diagnostics":
            self.app.export_diagnostics()
        elif what == "keyboard":
            permissions.open_url(permissions.KEYBOARD_SETTINGS)
        elif what == "login":
            login_item.open_system_settings()
        elif what == "mic":
            permissions.open_url(permissions.MICROPHONE_SETTINGS)
        elif what == "a11y":
            permissions.open_url(permissions.ACCESSIBILITY_SETTINGS)
        elif what == "settings":
            self.show_settings("privacy")
        elif what == "openai-keys":
            permissions.open_url(trial.KEYS_URL)
        elif what in _open_urls():
            permissions.open_url(_open_urls()[what])

    # ----------------------------------------------------------- history

    def history_payload(self) -> dict:
        db = self._db()
        user_paths = getattr(self.app.config, "user_paths", None)
        items = history.recent_sessions(db, HISTORY_ROWS) if db is not None else []
        mine = vocab.load_user_terms(Path(user_paths.vocab_path)) if user_paths is not None else []
        return {"enabled": self.app.prefs.save_history, "items": items, "vocab": mine, "words": history.list_words(db) if db is not None else []}

    def _history_message(self, msg: dict) -> None:
        kind = msg.get("t")
        window = self.history
        if kind == "ready":
            self._env(window)
            window.send({"t": "items", **self.history_payload()})
        elif kind == "geo":
            window.apply_glass(msg.get("s") or [])
        elif kind == "copy":
            text = str(msg.get("text") or "")
            if text:
                set_clipboard_text(text)
                window.send({"t": "toast", "msg": t("已复制", "Copied")})
        elif kind == "delete":
            db = self._db()
            if db is not None and history.delete_session(db, int(msg.get("id") or 0)):
                self.refresh(history=True)
        elif kind == "delword":
            db = self._db()
            if db is not None and history.delete_word(db, int(msg.get("id") or 0)):
                self.refresh(history=True)
        elif kind == "vocab":
            term = str(msg.get("term") or "").strip()
            if term:
                self._edit_vocab(None, "add", term)
                window.send({"t": "items", **self.history_payload()})
                window.send({"t": "toast", "msg": t(f"已把“{term}”加入词库", f"Added “{term}” to the vocabulary")})
        elif kind == "open":
            self._open(str(msg.get("what") or ""))

    # -------------------------------------------------------- onboarding

    def onboarding_state(self) -> dict:
        app = self.app
        refine = app.config.refine
        download = getattr(app, "_download", None)
        model_ready = download is None and self._model_cached()
        return {
            "t": "state",
            "name": brand.display_name(),
            "mic": permissions.microphone_status(),
            "input": getattr(app.config, "input_device", "") or _current_input(),
            "ax": bool(has_accessibility_trust()),
            "key": (
                # On the free trial the guide offers the user's own OpenAI key, not the trial token.
                {"env": trial.OWN_KEY_ENV, "service": "OpenAI", "preset": refine.preset, "has": False, "trial": True}
                if refine.preset == trial.PRESET
                else {
                    "env": refine.api_key_env,
                    "service": service_name(refine.base_url),
                    "preset": refine.preset,
                    "has": bool(os.environ.get(refine.api_key_env or "")),
                    "trial": False,
                }
            ),
            "model": {
                "name": short_model(mlx_whisper_repo((app.config.jarvis_config or {}).get("asr") or {})),
                "ready": model_ready,
                # Not ready and not downloading: the download at launch failed
                # (offline) before the guide was open to hear about it.
                "downloading": download is not None,
                "p": round(download[0], 3) if download else (1.0 if model_ready else 0.0),
                "mirror": reach.model_endpoint(app.prefs.model_source) == reach.MIRROR,
            },
        }

    def _model_cached(self) -> bool:
        from typeless_local.first_run import model_is_cached  # noqa: PLC0415

        asr_config = (self.app.config.jarvis_config or {}).get("asr") or {}
        if str(asr_config.get("provider") or "").strip().lower() != "mlx_whisper":
            return True
        return model_is_cached(mlx_whisper_repo(asr_config))

    def _onboarding_message(self, msg: dict) -> None:
        kind = msg.get("t")
        window = self.onboarding
        if kind == "ready":
            self._env(window)
            window.send(self.onboarding_state())
        elif kind == "mic":
            if permissions.microphone_status() == "not_determined":
                self._run(permissions.request_microphone, "ask-microphone")
                self._poll("mic", lambda: permissions.microphone_status() != "not_determined")
            else:
                permissions.open_url(permissions.MICROPHONE_SETTINGS)
                self._poll("mic", lambda: permissions.microphone_status() == "authorized")
        elif kind == "a11y":
            request_accessibility_trust()
            permissions.open_url(permissions.ACCESSIBILITY_SETTINGS)
            self._poll("ax", has_accessibility_trust)
        elif kind == "key":
            self._save_key(window, str(msg.get("env") or ""), str(msg.get("value") or ""), test=True)
        elif kind == "download":
            if getattr(self.app, "_download", None) is None and not self._model_cached():
                self.app._start_model_prefetch()
        elif kind == "lang":
            self._apply_setting("ui_language", str(msg.get("v") or ""))
        elif kind == "source":
            self._apply_setting("model_source", str(msg.get("v") or ""))
            if getattr(self.app, "_download", None) is None and not self._model_cached():
                self.app._start_model_prefetch()
        elif kind == "done":
            try:
                # Written out so a later launch never takes this for an install from before 0.4.0.
                self.app.set_preference("f5_hotkey", self.app.prefs.f5_hotkey)
                self.app.set_preference("onboarding_done", True)
            except Exception:
                LOGGER.exception("Could not record that onboarding finished")
            window.close()
        elif kind == "open":
            self._open(str(msg.get("what") or ""))
        elif kind == "meter":
            self._set_meter(bool(msg.get("on")))

    def guide(self, message: dict) -> None:
        """Something the guide reacts to happened elsewhere: a key pressed, a dictation finished."""

        if self._visible("onboarding"):
            self.onboarding.send(message)

    def _set_meter(self, on: bool) -> None:
        """The microphone page's live level: the real input, only while that page is up."""

        if on and self._meter is not None:
            return  # already listening: the guide asks again after every redraw
        stream, self._meter = self._meter, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                LOGGER.debug("Could not close the level stream", exc_info=True)
        if not on or permissions.microphone_status() != "authorized":
            return
        last = [0.0]

        def callback(indata, frames, time_info, status) -> None:
            now = time.monotonic()
            if now - last[0] < METER_EVERY_S:
                return
            last[0] = now
            peak = float(abs(indata).max()) if frames else 0.0
            self.app._call_ui(self._send_level, round(min(1.0, peak), 3))

        try:
            import sounddevice  # noqa: PLC0415

            stream = sounddevice.InputStream(
                device=devices.resolve_input_index(getattr(self.app.config, "input_device", "")),
                channels=1, samplerate=16000, callback=callback,
            )
            stream.start()
        except Exception:
            LOGGER.warning("No level meter for the guide", exc_info=True)
            return
        self._meter = stream

    def _send_level(self, value: float) -> None:
        if self._visible("onboarding"):
            self.onboarding.send({"t": "level", "v": value})
        else:
            self._set_meter(False)  # closed without saying so: let the microphone go

    def _poll(self, name: str, done: Callable[[], bool]) -> None:
        """Watch a permission while the guide waits on it, and redraw when it changes."""

        if name in self._polls:
            return
        self._polls.add(name)

        def job() -> None:
            deadline = time.monotonic() + POLL_TIMEOUT_S
            try:
                while time.monotonic() < deadline and self._visible("onboarding"):
                    if done():
                        break
                    time.sleep(POLL_S)
                self.app._call_ui(self._push_onboarding)
            finally:
                self._polls.discard(name)

        self._run(job, f"poll-{name}")

    def _push_onboarding(self) -> None:
        if self._visible("onboarding"):
            self.onboarding.send(self.onboarding_state())


def _current_input() -> str:
    try:
        return devices.current_input_device()
    except Exception:
        return ""


# ------------------------------------------------------------------ AppKit

_WINDOW_TITLED = 1 << 0
_WINDOW_CLOSABLE = 1 << 1
_WINDOW_MINIATURIZABLE = 1 << 2
_WINDOW_RESIZABLE = 1 << 3
_WINDOW_FULL_SIZE_CONTENT = 1 << 15
_BACKING_BUFFERED = 2
_TITLE_HIDDEN = 1
_TOOLBAR_UNIFIED = 3
_SEPARATOR_NONE = 1
DRAG_STRIP_H = 32.0
# An empty unified toolbar makes the title bar this tall and moves the traffic
# lights in from the corner, so they sit inside the sidebar card with even
# margins, and the window takes the rounder corners a card nests inside.
TOOLBAR_BAND_H = 52.0


class WebWindow:
    """A titled window whose content is one page; built the first time it is shown."""

    def __init__(
        self,
        page: str,
        title: str,
        size: tuple[float, float],
        on_message: Callable[[dict], None],
        min_size: tuple[float, float] | None = None,
        resizable: bool = True,
        toolbar: bool = False,
    ) -> None:
        self.page_name = page
        self.title = title
        self.size = size
        self.min_size = min_size
        self.resizable = resizable
        self.toolbar = toolbar
        self.on_message = on_message
        self.window = None
        self.page = None
        self.glass = None
        self.has_glass = False
        self.shown = False
        self._ready = False
        self._queue: list[dict] = []

    def _build(self) -> None:
        from AppKit import NSColor, NSWindow  # noqa: PLC0415
        from Foundation import NSMakeRect, NSMakeSize  # noqa: PLC0415

        from typeless_local.glass import GlassLayer  # noqa: PLC0415
        from typeless_local.webview import WebPage  # noqa: PLC0415

        width, height = self.size
        style = _WINDOW_TITLED | _WINDOW_CLOSABLE | _WINDOW_MINIATURIZABLE | _WINDOW_FULL_SIZE_CONTENT
        if self.resizable:
            style |= _WINDOW_RESIZABLE
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, width, height), style, _BACKING_BUFFERED, False
        )
        window.setTitle_(self.title)
        window.setTitlebarAppearsTransparent_(True)
        window.setTitleVisibility_(_TITLE_HIDDEN)
        window.setReleasedWhenClosed_(False)
        window.setBackgroundColor_(NSColor.windowBackgroundColor())
        if self.min_size:
            window.setContentMinSize_(NSMakeSize(*self.min_size))
        if self.toolbar:
            self._add_toolbar(window)
        content = window.contentView()
        bounds = content.bounds()
        self.glass = GlassLayer(content)
        self.has_glass = bool(self.glass.kind)
        self.page = WebPage(bounds, self._on_page_message)
        self.page.view.setAutoresizingMask_(18)  # width + height sizable
        content.addSubview_(self.page.view)
        # The page covers the title bar; this strip above it moves the window.
        strip_h = TOOLBAR_BAND_H if self.toolbar else DRAG_STRIP_H
        strip = _drag_view_class().alloc().initWithFrame_(
            NSMakeRect(0, bounds.size.height - strip_h, bounds.size.width, strip_h)
        )
        strip.setAutoresizingMask_(2 | 8)  # width sizable + flexible bottom margin
        content.addSubview_(strip)
        self._delegate = _window_delegate(self)
        window.setDelegate_(self._delegate)
        # Centred the first time; after that where the user left it.
        window.center()
        window.setFrameAutosaveName_(f"{brand.BUNDLE_ID}.{self.page_name}")
        self.window = window
        self.page.load(self.page_name)

    def _add_toolbar(self, window) -> None:
        try:
            from AppKit import NSToolbar  # noqa: PLC0415

            toolbar = NSToolbar.alloc().initWithIdentifier_(f"{brand.BUNDLE_ID}.{self.page_name}.toolbar")
            window.setToolbar_(toolbar)
            window.setToolbarStyle_(_TOOLBAR_UNIFIED)
            window.setTitlebarSeparatorStyle_(_SEPARATOR_NONE)
        except Exception:
            LOGGER.debug("No unified toolbar; the traffic lights stay in the corner", exc_info=True)

    def show(self) -> None:
        from AppKit import NSApplication  # noqa: PLC0415

        if self.window is None:
            self._build()
        self.shown = True
        _OPEN.add(self)
        _set_regular(True)
        self.window.makeKeyAndOrderFront_(None)
        # Keys go to the page (Enter, Cmd+1…7, the arrows) without a click first.
        self.window.makeFirstResponder_(self.page.view)
        # A menu-bar app's window only gets the keyboard once the app is active.
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def set_title(self, title: str) -> None:
        self.title = title
        if self.window is not None:
            self.window.setTitle_(title)

    def visible(self) -> bool:
        """Shown and not closed since; safe to ask from any thread."""

        return self.shown

    def close(self) -> None:
        if self.window is not None:
            self.window.performClose_(None)

    def send(self, message: dict) -> None:
        if self.page is None:
            return
        if not self._ready:
            self._queue.append(message)
            return
        self.page.send(message)

    def apply_glass(self, shapes: list[dict]) -> None:
        if self.glass is not None:
            self.glass.apply(shapes)

    def suppress_glass(self, suppressed: bool) -> None:
        if self.glass is not None:
            self.glass.set_suppressed(suppressed)

    def _on_page_message(self, message: dict) -> None:
        if message.get("t") == "ready":
            # A reload (the web process was restarted) asks again from scratch.
            self._ready = True
            queued, self._queue = self._queue, []
            self.on_message(message)
            # What the ready handler already resent fresh is not replayed.
            for item in queued:
                if item.get("t") not in ("env", "state", "items"):
                    self.page.send(item)
            return
        self.on_message(message)

    def will_close(self) -> None:
        """The page stays loaded while hidden, so reopening is instant."""

        self.shown = False
        _OPEN.discard(self)
        if not _OPEN:
            _set_regular(False)


_DragView = None
_Delegate = None
_DELEGATES: list = []
_OPEN: set = set()
_POLICY_REGULAR = 0
_POLICY_ACCESSORY = 1


def _set_regular(regular: bool) -> None:
    """A Dock icon and a place in Cmd+Tab while a window is open; menu bar only otherwise."""

    try:
        from AppKit import NSApplication  # noqa: PLC0415

        NSApplication.sharedApplication().setActivationPolicy_(_POLICY_REGULAR if regular else _POLICY_ACCESSORY)
    except Exception:
        LOGGER.debug("Could not change the activation policy", exc_info=True)


def _drag_view_class():
    global _DragView
    if _DragView is None:
        from AppKit import NSView  # noqa: PLC0415

        # Objective-C class names are process-wide, hence the prefix.
        class YanaDragView(NSView):
            def mouseDown_(self, event) -> None:  # noqa: N802 - Cocoa selector
                window = self.window()
                if window is None:
                    return
                if event.clickCount() == 2:
                    window.performZoom_(None)
                    return
                window.performWindowDragWithEvent_(event)

        _DragView = YanaDragView
    return _DragView


def _window_delegate(owner: WebWindow):
    global _Delegate
    if _Delegate is None:
        import objc  # noqa: PLC0415
        from Foundation import NSObject  # noqa: PLC0415

        class YanaWindowDelegate(NSObject):
            def initWithOwner_(self, owner_):  # noqa: N802
                self = objc.super(YanaWindowDelegate, self).init()
                if self is None:
                    return None
                self.owner = owner_
                return self

            def windowWillClose_(self, notification) -> None:  # noqa: N802
                self.owner.will_close()

        _Delegate = YanaWindowDelegate
    delegate = _Delegate.alloc().initWithOwner_(owner)
    _DELEGATES.append(delegate)
    return delegate


def install_main_menu() -> None:
    """Edit and window commands for the windows' text fields.

    The app is an accessory with no visible menu bar, but Cmd+C / Cmd+V / Cmd+Z
    in a web page only work when a main-menu item carries them.
    """

    try:
        from AppKit import NSApplication, NSMenu, NSMenuItem  # noqa: PLC0415
    except Exception:
        return
    app = NSApplication.sharedApplication()
    main = NSMenu.alloc().initWithTitle_("")

    def submenu(title: str, items: list[tuple[str, str, str, int]]) -> None:
        holder = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
        menu = NSMenu.alloc().initWithTitle_(title)
        for label, action, key, mask in items:
            if not label:
                menu.addItem_(NSMenuItem.separatorItem())
                continue
            item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(label, action, key)
            item.setKeyEquivalentModifierMask_(mask)
            menu.addItem_(item)
        holder.setSubmenu_(menu)
        main.addItem_(holder)

    command, shift = 1 << 20, 1 << 17
    submenu(
        brand.display_name(),
        [(t("关闭窗口", "Close Window"), "performClose:", "w", command), ("", "", "", 0), (brand.quit_label(), "terminate:", "q", command)],
    )
    submenu(
        t("编辑", "Edit"),
        [
            (t("撤销", "Undo"), "undo:", "z", command),
            (t("重做", "Redo"), "redo:", "z", command | shift),
            ("", "", "", 0),
            (t("剪切", "Cut"), "cut:", "x", command),
            (t("拷贝", "Copy"), "copy:", "c", command),
            (t("粘贴", "Paste"), "paste:", "v", command),
            (t("全选", "Select All"), "selectAll:", "a", command),
        ],
    )
    app.setMainMenu_(main)
