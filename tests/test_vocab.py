from __future__ import annotations

from pathlib import Path


from typeless_local import vocab


def test_load_vocab_missing_file_returns_empty_list(tmp_path: Path) -> None:
    result = vocab.load_vocab(tmp_path / "missing.yaml")
    assert result == []


def test_load_vocab_combines_user_and_auto_with_user_first(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text(
        "user:\n  - Jarvis\n  - Typeless\n"
        "auto:\n  - Hermes Aging\n  - PyObjC\n",
        encoding="utf-8",
    )
    result = vocab.load_vocab(path)
    assert result == ["Jarvis", "Typeless", "Hermes Aging", "PyObjC"]


def test_load_vocab_dedups_term_in_both_user_and_auto(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text(
        "user:\n  - Jarvis\nauto:\n  - Jarvis\n  - Typeless\n",
        encoding="utf-8",
    )
    result = vocab.load_vocab(path)
    assert result == ["Jarvis", "Typeless"]


def test_load_vocab_malformed_yaml_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text("not: valid: yaml: [", encoding="utf-8")
    result = vocab.load_vocab(path)
    assert result == []


def test_as_initial_prompt_empty_returns_empty_string() -> None:
    assert vocab.as_initial_prompt([]) == ""


def test_as_initial_prompt_renders_comma_separated() -> None:
    result = vocab.as_initial_prompt(["Jarvis", "Typeless"])
    assert result == "Common terms: Jarvis, Typeless."


def test_as_initial_prompt_truncates_at_term_boundary() -> None:
    terms = ["A" * 100 for _ in range(20)]
    out = vocab.as_initial_prompt(terms, max_chars=300)
    # Should fit under budget and end with '.'
    assert len(out) <= 300
    assert out.endswith(".")
    # First term must always be present.
    assert "A" * 100 in out


def test_save_auto_terms_preserves_user_section(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text(
        "user:\n  - Jarvis\n  - Typeless\nauto:\n  - Old\n",
        encoding="utf-8",
    )
    vocab.save_auto_terms(path, ["NewTerm", "AnotherTerm"])
    loaded = vocab.load_vocab(path)
    assert loaded == ["Jarvis", "Typeless", "NewTerm", "AnotherTerm"]


def test_save_auto_terms_creates_file_if_missing(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    vocab.save_auto_terms(path, ["Term1"])
    loaded = vocab.load_vocab(path)
    assert loaded == ["Term1"]


def test_load_rejected_missing_file_returns_empty(tmp_path: Path) -> None:
    assert vocab.load_rejected(tmp_path / "missing.yaml") == []


def test_load_rejected_reads_explicit_section(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text(
        "user:\n  - Jarvis\nauto: []\nrejected:\n  - Hermes Aging\n  - Foo Bar\n",
        encoding="utf-8",
    )
    assert vocab.load_rejected(path) == ["Hermes Aging", "Foo Bar"]


def test_save_auto_terms_preserves_rejected_section(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text(
        "user:\n  - Jarvis\nauto:\n  - Old\nrejected:\n  - Hermes Aging\n",
        encoding="utf-8",
    )
    vocab.save_auto_terms(path, ["NewTerm"])
    assert vocab.load_rejected(path) == ["Hermes Aging"]
    assert vocab.load_vocab(path) == ["Jarvis", "NewTerm"]


def test_reject_term_moves_term_from_auto_into_rejected(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text(
        "user:\n  - Jarvis\nauto:\n  - Hermes Aging\n  - PyObjC\nrejected: []\n",
        encoding="utf-8",
    )
    vocab.reject_term(path, "Hermes Aging")
    assert vocab.load_vocab(path) == ["Jarvis", "PyObjC"]
    assert vocab.load_rejected(path) == ["Hermes Aging"]


def test_reject_term_is_case_insensitive_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text(
        "user: []\nauto:\n  - Hermes Aging\nrejected: []\n",
        encoding="utf-8",
    )
    vocab.reject_term(path, "hermes aging")
    vocab.reject_term(path, "Hermes Aging")
    assert vocab.load_rejected(path) == ["hermes aging"]
    assert vocab.load_vocab(path) == []


def test_reject_term_creates_file_if_missing(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    vocab.reject_term(path, "Mishear")
    assert vocab.load_rejected(path) == ["Mishear"]


def test_load_user_terms_leaves_out_auto_terms(tmp_path) -> None:
    from typeless_local.vocab import load_user_terms

    path = tmp_path / "vocab.yaml"
    path.write_text("user:\n- Jarvis\n- StarTrial\nauto:\n- mishear\n", encoding="utf-8")

    assert load_user_terms(path) == ["Jarvis", "StarTrial"]


def test_save_user_terms_keeps_auto_and_rejected(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text("user: [Old]\nauto: [Auto]\nrejected: [Nope]\n", encoding="utf-8")

    vocab.save_user_terms(path, ["Typlus", " PyObjC ", "", "Typlus"])

    assert vocab.load_user_terms(path) == ["Typlus", "PyObjC"]
    assert vocab.load_vocab(path) == ["Typlus", "PyObjC", "Auto"]
    assert vocab.load_rejected(path) == ["Nope"]


def test_learn_unlearn_round_trip_and_auto_save_keeps_learned(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    assert vocab.learn_term(path, "merge", "默制") is True
    assert vocab.learn_term(path, "Merge", "默治") is True  # counted again under the kept spelling
    vocab.save_auto_terms(path, ["PyObjC"])
    sections = vocab._load_sections(path)
    assert sections["user"] == ["merge"] and sections["auto"] == ["PyObjC"]
    assert [(e["term"], e["was"], e["n"]) for e in sections["learned"]] == [("merge", "默治", 2)] and sections["learned"][0]["at"]
    vocab.unlearn_term(path, "merge")
    sections = vocab._load_sections(path)
    assert sections["user"] == [] and sections["learned"] == [] and sections["rejected"] == ["merge"]
    assert vocab.learn_term(path, "merge", "默制") is False


def test_whisper_terms_rank_fixes_then_use_then_recency(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    vocab.save_user_terms(path, ["Jarvis", "Typlus"])
    vocab.learn_term(path, "merge", "默制")
    vocab.learn_term(path, "Jev", "Jeff")
    assert vocab.whisper_terms(path) == ["Jev", "merge", "Jarvis", "Typlus"]
    # Plain use counts, but one fix weighs more; a repeat fix counts again.
    assert vocab.whisper_terms(path, {"Typlus": 3}) == ["Jev", "merge", "Typlus", "Jarvis"]
    assert vocab.whisper_terms(path, {"Typlus": 6}) == ["Typlus", "Jev", "merge", "Jarvis"]
    assert vocab.learn_term(path, "merge", "默治")
    assert vocab.whisper_terms(path, {"Typlus": 6}) == ["merge", "Typlus", "Jev", "Jarvis"]
    assert vocab.learn_term(path, "jarvis", "Java")  # a hand-added word fixed by hand counts too
    assert vocab.load_user_terms(path) == ["Jarvis", "Typlus", "merge", "Jev"]
    assert [(e["term"], e["n"]) for e in vocab._load_sections(path)["learned"]] == [("Jev", 1), ("merge", 2), ("Jarvis", 1)]


def test_hotword_weights_favour_words_fixed_by_hand(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    vocab.save_user_terms(path, ["Jarvis"])
    vocab.save_auto_terms(path, ["PyObjC"])
    vocab.learn_term(path, "merge", "默制")
    vocab.learn_term(path, "Jev", "Jeff")
    vocab.learn_term(path, "Jev", "Java")

    assert vocab.hotword_weights(path) == {"PyObjC": 3, "Jarvis": 4, "merge": 5, "Jev": 50}
