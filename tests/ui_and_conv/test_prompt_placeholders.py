from __future__ import annotations

from PIL import Image

from kimi_cli.ui.shell import placeholders
from kimi_cli.ui.shell.placeholders import (
    AttachmentCache,
    PromptPlaceholderManager,
    should_placeholderize_pasted_text,
)
from kimi_cli.wire.types import ImageURLPart, TextPart


def test_placeholder_manager_serializes_text_tokens_for_history(tmp_path) -> None:
    manager = PromptPlaceholderManager(attachment_cache=AttachmentCache(root=tmp_path))
    text_token = manager.maybe_placeholderize_pasted_text("alpha\nbeta\ngamma")
    image = Image.new("RGB", (4, 4), color=(10, 20, 30))
    image_token = manager.create_image_placeholder(image)

    assert image_token is not None

    history_text = manager.serialize_for_history(f"before {text_token} {image_token} after")

    assert history_text == f"before alpha\nbeta\ngamma {image_token} after"


def test_placeholder_manager_refolds_editor_text_for_known_text_tokens() -> None:
    manager = PromptPlaceholderManager()
    text_token = manager.maybe_placeholderize_pasted_text("alpha\nbeta\ngamma")
    original_command = f"before {text_token} after"

    refolded = manager.refold_after_editor(
        "before alpha\nbeta\ngamma after\nnotes", original_command
    )

    assert refolded == f"before {text_token} after\nnotes"


def test_placeholder_manager_refolds_original_placeholder_span_not_first_duplicate() -> None:
    manager = PromptPlaceholderManager()
    pasted_text = "alpha\nbeta\ngamma"
    text_token = manager.maybe_placeholderize_pasted_text(pasted_text)
    original_command = f"{pasted_text}\n---\n{text_token}"

    refolded = manager.refold_after_editor(f"{pasted_text}\n---\n{pasted_text}", original_command)

    assert refolded == f"{pasted_text}\n---\n{text_token}"


def test_placeholder_manager_does_not_refold_moved_pasted_text() -> None:
    manager = PromptPlaceholderManager()
    pasted_text = "alpha\nbeta\ngamma"
    text_token = manager.maybe_placeholderize_pasted_text(pasted_text)
    original_command = f"{pasted_text}\n---\n{text_token}"
    edited_text = f"{pasted_text}\n{pasted_text}\n---\n"

    refolded = manager.refold_after_editor(edited_text, original_command)

    assert refolded == edited_text


def test_placeholder_manager_refolds_multiple_unedited_placeholders() -> None:
    manager = PromptPlaceholderManager()
    first = "alpha\nbeta\ngamma"
    second = "one\ntwo\nthree"
    first_token = manager.maybe_placeholderize_pasted_text(first)
    second_token = manager.maybe_placeholderize_pasted_text(second)
    original_command = f"{first_token}\n---\n{second_token}"

    refolded = manager.refold_after_editor(f"{first}\n---\n{second}", original_command)

    assert refolded == original_command


def test_placeholder_manager_only_refolds_unedited_placeholder_when_multiple_exist() -> None:
    manager = PromptPlaceholderManager()
    first = "alpha\nbeta\ngamma"
    second = "one\ntwo\nthree"
    first_token = manager.maybe_placeholderize_pasted_text(first)
    second_token = manager.maybe_placeholderize_pasted_text(second)
    original_command = f"{first_token}\n---\n{second_token}"

    refolded = manager.refold_after_editor(
        f"{first}\n---\none\ntwo changed\nthree", original_command
    )

    assert refolded == f"{first_token}\n---\none\ntwo changed\nthree"


def test_placeholder_manager_leaves_unknown_text_token_literal() -> None:
    manager = PromptPlaceholderManager()

    resolved = manager.resolve_command("[Pasted text #999 +3 lines]")

    assert resolved.resolved_text == "[Pasted text #999 +3 lines]"
    assert resolved.content == [TextPart(text="[Pasted text #999 +3 lines]")]


def test_placeholder_manager_resolves_mixed_text_and_image_tokens(tmp_path) -> None:
    manager = PromptPlaceholderManager(attachment_cache=AttachmentCache(root=tmp_path))
    pasted_text = "\n".join([f"line{i}" for i in range(1, 16)])
    text_token = manager.maybe_placeholderize_pasted_text(pasted_text)
    image = Image.new("RGB", (4, 4), color=(10, 20, 30))
    image_token = manager.create_image_placeholder(image)

    assert image_token is not None

    resolved = manager.resolve_command(f"look {text_token} {image_token}")

    assert resolved.resolved_text == f"look {pasted_text} {image_token}"
    assert resolved.content[0] == TextPart(text="look ")
    assert resolved.content[1] == TextPart(text=pasted_text)
    assert resolved.content[2] == TextPart(text=" ")
    assert resolved.content[3].type == "text"
    assert resolved.content[4].type == "image_url"
    assert isinstance(resolved.content[4], ImageURLPart)
    assert resolved.content[5] == TextPart(text="</image>")


def test_placeholder_manager_expands_text_but_not_image_for_editor(tmp_path) -> None:
    manager = PromptPlaceholderManager(attachment_cache=AttachmentCache(root=tmp_path))
    text_token = manager.maybe_placeholderize_pasted_text("alpha\nbeta\ngamma")
    image = Image.new("RGB", (4, 4), color=(10, 20, 30))
    image_token = manager.create_image_placeholder(image)

    assert image_token is not None

    editor_text = manager.expand_for_editor(f"before {text_token} {image_token} after")

    assert editor_text == f"before alpha\nbeta\ngamma {image_token} after"


def test_placeholder_manager_leaves_unknown_image_placeholder_literal() -> None:
    manager = PromptPlaceholderManager()

    resolved = manager.resolve_command("[image:missing.png,10x10]")

    assert resolved.resolved_text == "[image:missing.png,10x10]"
    assert resolved.content == [TextPart(text="[image:missing.png,10x10]")]


def test_placeholder_manager_sanitizes_surrogates_in_pasted_text() -> None:
    manager = PromptPlaceholderManager()
    # Lone surrogate \ud83d (half of an emoji pair) must not survive into the entry.
    text_with_surrogate = "A" * 1000 + "\ud83d"
    token = manager.maybe_placeholderize_pasted_text(text_with_surrogate)

    resolved = manager.resolve_command(token)

    # The surrogate must not survive; it is replaced with U+FFFD characters.
    assert "\ud83d" not in resolved.resolved_text
    assert resolved.resolved_text.startswith("A" * 1000)
    assert "\ufffd" in resolved.resolved_text

    # Serialization for history must not raise.
    history = manager.serialize_for_history(token)
    assert "\ud83d" not in history


def test_placeholderize_thresholds_cover_char_and_line_boundaries() -> None:
    assert should_placeholderize_pasted_text("A" * 999) is False
    assert should_placeholderize_pasted_text("A" * 1000) is True
    assert should_placeholderize_pasted_text("line1\nline2") is False
    assert should_placeholderize_pasted_text("\n".join([f"line{i}" for i in range(1, 15)])) is False
    assert should_placeholderize_pasted_text("\n".join([f"line{i}" for i in range(1, 16)])) is True


def test_placeholder_manager_normalizes_crlf_before_threshold_and_resolution() -> None:
    manager = PromptPlaceholderManager()
    lines = "\r\n".join([f"line{i}" for i in range(1, 16)])
    token = manager.maybe_placeholderize_pasted_text(lines)

    assert token == "[Pasted text #1 +15 lines]"

    resolved = manager.resolve_command(token)
    assert resolved.resolved_text == "\n".join([f"line{i}" for i in range(1, 16)])


def test_placeholderize_thresholds_are_configurable(monkeypatch) -> None:
    monkeypatch.setattr(placeholders, "_TEXT_PASTE_CHAR_THRESHOLD", 50)
    monkeypatch.setattr(placeholders, "_TEXT_PASTE_LINE_THRESHOLD", 3)

    assert should_placeholderize_pasted_text("A" * 49) is False
    assert should_placeholderize_pasted_text("A" * 50) is True
    assert should_placeholderize_pasted_text("a\nb") is False
    assert should_placeholderize_pasted_text("a\nb\nc") is True


def test_get_env_int_parses_valid_values(monkeypatch) -> None:
    from kimi_cli.utils.envvar import get_env_int

    monkeypatch.setenv("_TEST_INT_VAR", "42")
    assert get_env_int("_TEST_INT_VAR", 0) == 42


def test_get_env_int_falls_back_on_invalid_values(monkeypatch) -> None:
    from kimi_cli.utils.envvar import get_env_int

    monkeypatch.setenv("_TEST_INT_VAR", "not_a_number")
    assert get_env_int("_TEST_INT_VAR", 99) == 99


def test_get_env_int_returns_default_when_unset() -> None:
    from kimi_cli.utils.envvar import get_env_int

    assert get_env_int("_TEST_NONEXISTENT_VAR_12345", 77) == 77


def test_find_next_match_agrees_with_naive_two_handler_scan() -> None:
    """The single-pass scan must find exactly what the per-handler scans find.

    Guards the union invariant of `_PLACEHOLDER_SCAN_RE`: every candidate it
    reports must be attributable to exactly one handler at the same span.
    """
    from kimi_cli.ui.shell.placeholders import (
        _IMAGE_PLACEHOLDER_RE,
        _PASTED_TEXT_PLACEHOLDER_RE,
        _PLACEHOLDER_SCAN_RE,
    )

    corpus = [
        "",
        "plain text without brackets",
        "[Pasted text #1]",
        "[Pasted text #1 +1 line]",
        "[Pasted text #12 +34 lines]",
        "[Pasted text #1 +0 lines][Pasted text #2]",
        "[image:abc123.png,10x20]",
        "[image:abc123.png]",
        "[img:x.y-z_0,1x2]",
        "[Pasted text #1][image:a.png,2x3][Pasted text #2 +4 lines]",
        "[image:a.png][image:b.png,1x1]",
        "[not a token]",
        "[Pasted text #]",
        "[Pasted text #1 + lines]",
        "[Pasted text #1 +2 linez]",
        "[[Pasted text #1]]",
        "[[image:a.png]]",
        "[Pasted text #1",
        "Pasted text #1]",
        "[image:]",
        "[image:a.png,]",
        "[image:a.png,10x]",
        "[IMAGE:A.PNG]",
        "[pasted text #1]",
        "[Pasted text  #1]",
        "x" * 5000 + "[Pasted text #7 +8 lines]" + "y" * 5000,
        "[a:b]" * 100,
        "[Pasted text #1 +1 line]" * 100,
        "mix [Pasted text #3] middle [image:z.png,9x9] end",
    ]
    for text in corpus:
        naive_spans: list[tuple[int, int, str]] = []
        pos = 0
        while True:
            earliest: tuple[int, int, str] | None = None
            for pattern, kind in (
                (_PASTED_TEXT_PLACEHOLDER_RE, "text"),
                (_IMAGE_PLACEHOLDER_RE, "image"),
            ):
                found = pattern.search(text, pos)
                if found is not None and (earliest is None or found.start() < earliest[0]):
                    earliest = (found.start(), found.end(), kind)
            if earliest is None:
                break
            naive_spans.append(earliest)
            pos = earliest[1]

        scan_spans: list[tuple[int, int, str]] = []
        pos = 0
        while (scan := _PLACEHOLDER_SCAN_RE.search(text, pos)) is not None:
            attributed: tuple[int, int, str] | None = None
            for pattern, kind in (
                (_PASTED_TEXT_PLACEHOLDER_RE, "text"),
                (_IMAGE_PLACEHOLDER_RE, "image"),
            ):
                anchored = pattern.match(text, scan.start())
                if anchored is not None:
                    attributed = (anchored.start(), anchored.end(), kind)
                    break
            assert attributed is not None, f"unattributable scan hit in {text!r}"
            assert (attributed[0], attributed[1]) == (
                scan.start(),
                scan.end(),
            ), f"span drift in {text!r}"
            scan_spans.append(attributed)
            pos = attributed[1]

        assert scan_spans == naive_spans, f"mismatch in {text!r}"


def test_resolve_command_scales_linearly_with_many_tokens() -> None:
    """Regression test for issue #2655: resolving thousands of placeholder-like
    tokens in a large prompt must not hang (was O(tokens x len), ~260s)."""
    import time

    manager = PromptPlaceholderManager()
    token_count = 20000
    command = "[Pasted text #1 +5 lines]" * token_count

    started = time.perf_counter()
    resolved = manager.resolve_command(command)
    elapsed = time.perf_counter() - started

    assert len(resolved.resolved_text) == len(command)
    assert elapsed < 10, f"resolve_command took {elapsed:.1f}s for {token_count} tokens"


def test_serialize_for_history_scales_with_tokens_and_unmatched_brackets() -> None:
    """Thousands of tokens plus unmatched-bracket noise must stay linear.

    The old per-handler scan re-scanned the whole tail for every token when
    one token kind was absent, hanging for tens of seconds here.
    """
    import time

    manager = PromptPlaceholderManager()
    command = "[Pasted text #1 +5 lines][oops " * 20000

    started = time.perf_counter()
    history = manager.serialize_for_history(command)
    elapsed = time.perf_counter() - started

    assert history == command  # unknown tokens stay literal
    assert elapsed < 10, f"serialize_for_history took {elapsed:.1f}s"


def test_refold_after_editor_skips_diff_for_unchanged_large_paste() -> None:
    """Regression test for issue #2655: refolding an unedited ~900KB paste
    must not run the superlinear SequenceMatcher diff (hung 15+ minutes)."""
    import time

    manager = PromptPlaceholderManager()
    pasted_text = "lorem ipsum dolor sit amet\n" * 36000  # ~900KB
    token = manager.maybe_placeholderize_pasted_text(pasted_text)
    original_command = f"Summarize this: {token}"
    expanded = manager.expand_for_editor(original_command)

    started = time.perf_counter()
    refolded = manager.refold_after_editor(expanded, original_command)
    elapsed = time.perf_counter() - started

    assert refolded == original_command
    assert elapsed < 10, f"refold_after_editor took {elapsed:.1f}s for unchanged paste"


def test_attachment_cache_loads_legacy_root(tmp_path) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_image_dir = legacy_root / "images"
    legacy_image_dir.mkdir(parents=True)
    attachment_id = "legacy.png"
    payload = b"\x89PNG\r\n\x1a\nlegacy"
    (legacy_image_dir / attachment_id).write_bytes(payload)

    cache = AttachmentCache(root=tmp_path / "new-root", legacy_roots=(legacy_root,))
    loaded = cache.load_bytes("image", attachment_id)

    assert loaded is not None
    path, image_bytes = loaded
    assert path == legacy_image_dir / attachment_id
    assert image_bytes == payload
