"""User-managed vocabulary store for Whisper hotwords and refine prompts."""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

LOGGER = logging.getLogger(__name__)

_STARTER_HEADER = """# Typlus vocabulary.
# Edit `user:` freely — those terms are never overwritten.
# `auto:` is rewritten by `scripts/extract_hotwords.py`; don't hand-edit it.
# `rejected:` lists mishears the extractor must never re-promote.
"""


def _read_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        LOGGER.warning("Failed to read vocab at %s: %s", path, exc)
        return {}
    if not isinstance(loaded, dict):
        LOGGER.warning("Vocab at %s is not a mapping; ignoring", path)
        return {}
    return loaded


def _coerce_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                out.append(stripped)
    return out


def load_vocab(path: Path) -> list[str]:
    """Return deduplicated terms: user first, then auto. Missing/malformed → []."""

    path = Path(path)
    if not path.exists():
        return []
    data = _read_yaml(path)
    user_terms = _coerce_list(data.get("user"))
    auto_terms = _coerce_list(data.get("auto"))
    seen: set[str] = set()
    result: list[str] = []
    for term in user_terms + auto_terms:
        if term in seen:
            continue
        seen.add(term)
        result.append(term)
    return result


def load_user_terms(path: Path) -> list[str]:
    """The hand-kept `user:` terms only; these, not `auto:`, go into Whisper's prompt."""

    return _load_sections(Path(path))["user"]


FIX_WEIGHT = 5  # one hand fix outweighs a few plain uses


def whisper_terms(path: Path, uses: dict[str, int] | None = None) -> list[str]:
    """The `user:` terms hottest first, so Whisper's capped prompt keeps the words that matter.

    Heat is FIX_WEIGHT per hand fix plus ``uses`` (recent dictations that had
    the word); ties go to the most recently learned, then to the list order.
    """

    sections = _load_sections(Path(path))
    # `learned:` is kept oldest first, so a later position means fixed more recently.
    learned = {e["term"]: (e["n"], i) for i, e in enumerate(sections["learned"])}
    uses = uses or {}

    # Both sorts are stable: newest-learned order survives inside each heat.
    newest = sorted(dict.fromkeys(sections["user"]), key=lambda term: learned.get(term, (0, -1))[1], reverse=True)
    return sorted(newest, key=lambda term: -(FIX_WEIGHT * learned.get(term, (0, -1))[0] + uses.get(term, 0)))


# Qwen's hot word weights: picked up on their own < kept by hand < fixed by hand
# once; fixed twice or more, a word he keeps being misheard on, is a "super" hot word.
AUTO_HOTWORD, USER_HOTWORD, FIXED_HOTWORD, SUPER_HOTWORD = 3, 4, 5, 50


def hotword_weights(path: Path) -> dict[str, int]:
    """How much each term should pull Qwen's recognition toward it."""

    sections = _load_sections(Path(path))
    weights = dict.fromkeys(sections["auto"], AUTO_HOTWORD)
    weights.update(dict.fromkeys(sections["user"], USER_HOTWORD))
    for entry in sections["learned"]:
        weights[entry["term"]] = SUPER_HOTWORD if entry["n"] >= 2 else FIXED_HOTWORD
    return weights


def load_rejected(path: Path) -> list[str]:
    """Return the case-preserved list of terms the extractor must skip."""

    path = Path(path)
    if not path.exists():
        return []
    return _coerce_list(_read_yaml(path).get("rejected"))


def _coerce_learned(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [
        {"term": str(i["term"]).strip(), "was": str(i.get("was") or "").strip(), "at": str(i.get("at") or ""), "n": int(i.get("n") or 1)}
        for i in value
        if isinstance(i, dict) and str(i.get("term") or "").strip()
    ]


def _load_sections(path: Path) -> dict[str, list]:
    """Read user/auto/rejected/learned sections, defaulting missing ones to []."""

    data = _read_yaml(path) if path.exists() else {}
    return {
        "user": _coerce_list(data.get("user")),
        "auto": _coerce_list(data.get("auto")),
        "rejected": _coerce_list(data.get("rejected")),
        "learned": _coerce_learned(data.get("learned")),
    }


def _atomic_write_sections(path: Path, sections: dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user": sections.get("user", []),
        "auto": sections.get("auto", []),
        "rejected": sections.get("rejected", []),
    }
    if sections.get("learned"):
        payload["learned"] = sections["learned"]
    rendered = _STARTER_HEADER + yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=".vocab-", suffix=".yaml.tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def as_initial_prompt(terms: list[str], max_chars: int = 600) -> str:
    """Render terms as a single-line Whisper initial prompt.

    Terms are joined comma-separated under a 'Common terms:' prefix. The
    output is truncated at term boundaries to stay under ``max_chars``. The
    first term is always preserved; subsequent terms are added until adding
    one more would exceed the budget.
    """

    if not terms:
        return ""
    prefix = "Common terms: "
    suffix = "."
    budget = max_chars - len(prefix) - len(suffix)
    kept: list[str] = []
    used = 0
    for term in terms:
        addition = (", " if kept else "") + term
        if kept and used + len(addition) > budget:
            break
        kept.append(term)
        used += len(addition)
    return prefix + ", ".join(kept) + suffix


def save_auto_terms(path: Path, terms: list[str]) -> None:
    """Atomically rewrite `auto:`. Preserves `user:` and `rejected:`."""

    path = Path(path)
    sections = _load_sections(path)
    sections["auto"] = [t for t in terms if t.strip()]
    _atomic_write_sections(path, sections)


def save_user_terms(path: Path, terms: list[str]) -> None:
    """Atomically rewrite `user:` (the Settings window's word list).

    Preserves `auto:` and `rejected:`; blanks and repeats are dropped.
    """

    path = Path(path)
    sections = _load_sections(path)
    seen: set[str] = set()
    kept: list[str] = []
    for term in terms:
        cleaned = str(term).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            kept.append(cleaned)
    sections["user"] = kept
    _atomic_write_sections(path, sections)


def reject_term(path: Path, term: str) -> None:
    """Move ``term`` out of ``auto:`` and into ``rejected:`` (case-insensitive).

    Idempotent — if the term is already rejected we just rewrite the file.
    Used by ``scripts/extract_hotwords.py --reject`` so the user can blocklist
    a mishear (e.g. "Hermes Aging") without hand-editing YAML.
    """

    path = Path(path)
    cleaned = term.strip()
    if not cleaned:
        return
    lc = cleaned.lower()
    sections = _load_sections(path)
    sections["auto"] = [t for t in sections["auto"] if t.lower() != lc]
    if not any(t.lower() == lc for t in sections["rejected"]):
        sections["rejected"].append(cleaned)
    _atomic_write_sections(path, sections)


def learn_term(path: Path, term: str, was: str) -> bool:
    """Note a word the user fixed by hand in `learned:` (counting repeat fixes) and keep it in `user:`.

    Does nothing (False) when the word was rejected.
    """

    path = Path(path)
    term = term.strip()
    sections = _load_sections(path)
    if not term or any(term.lower() == t.lower() for t in sections["rejected"]):
        return False
    kept = next((t for t in sections["user"] if t.lower() == term.lower()), None)
    if kept is None:
        sections["user"].append(term)
    term = kept or term
    entry = next((e for e in sections["learned"] if e["term"] == term), {"term": term, "n": 0})
    sections["learned"] = [e for e in sections["learned"] if e is not entry] + [entry]
    entry.update(was=was.strip(), at=datetime.now().isoformat(timespec="minutes"), n=entry["n"] + 1)
    _atomic_write_sections(path, sections)
    return True


def unlearn_term(path: Path, term: str) -> None:
    """Drop a learned word from `user:` and `learned:`, and reject it so it is never learned again."""

    path = Path(path)
    lc = term.strip().lower()
    if not lc:
        return
    sections = _load_sections(path)
    sections["user"] = [t for t in sections["user"] if t.lower() != lc]
    sections["learned"] = [e for e in sections["learned"] if e["term"].lower() != lc]
    if not any(t.lower() == lc for t in sections["rejected"]):
        sections["rejected"].append(term.strip())
    _atomic_write_sections(path, sections)


def write_starter_file(path: Path) -> None:
    """Create a starter vocab.yaml with empty lists and a header comment."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(
        _STARTER_HEADER + "user: []\nauto: []\nrejected: []\n",
        encoding="utf-8",
    )
