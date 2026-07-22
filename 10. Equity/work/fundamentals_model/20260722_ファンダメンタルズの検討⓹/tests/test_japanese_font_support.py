from __future__ import annotations

from matplotlib import font_manager

from stock_scoring_model.font_support import resolve_japanese_font, setup_japanese_matplotlib


def test_resolved_font_contains_japanese_glyphs() -> None:
    family, path = resolve_japanese_font()
    assert family
    assert path.is_file()
    charmap = font_manager.get_font(str(path)).get_charmap()
    for char in "日本語あいう漢字":
        assert ord(char) in charmap


def test_matplotlib_uses_resolved_family() -> None:
    family, path = setup_japanese_matplotlib()
    assert family
    assert path.is_file()
