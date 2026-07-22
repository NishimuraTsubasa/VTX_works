from __future__ import annotations

import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from matplotlib import font_manager, rcParams

# The project never distributes font files.  The resolver only uses fonts that
# already exist on the execution PC, or an explicit path supplied by the user.
_ENV_KEYS = (
    "STOCK_SCORING_JAPANESE_FONT",
    "STOCK_SCORING_JP_FONT",
    "MPL_JAPANESE_FONT",
)

# A small glyph set that distinguishes a genuinely Japanese-capable font from
# a Latin-only font.  Checking the cmap prevents silent fallback to DejaVu Sans.
_REQUIRED_GLYPHS = "日本語あいう漢字ー（）"

_PREFERRED_FAMILIES = (
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "Yu Gothic",
    "YuGothic",
    "Meiryo",
    "BIZ UDPGothic",
    "BIZ UDゴシック",
    "MS Gothic",
    "IPAexGothic",
    "IPAGothic",
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "TakaoGothic",
)


def _common_font_paths() -> list[Path]:
    paths: list[Path] = []

    # Windows Japanese fonts.  WINDIR/SystemRoot is used instead of assuming C:.
    win_root = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if win_root:
        font_dir = Path(win_root) / "Fonts"
        paths.extend(
            font_dir / name
            for name in (
                "YuGothM.ttc",
                "YuGothR.ttc",
                "YuGothB.ttc",
                "meiryo.ttc",
                "meiryob.ttc",
                "BIZ-UDGothicR.ttc",
                "BIZ-UDGothicB.ttc",
                "msgothic.ttc",
                "NotoSansCJK-Regular.ttc",
                "NotoSansJP-Regular.ttf",
            )
        )

    # Linux paths used by common Ubuntu/RHEL distributions.
    paths.extend(
        Path(name)
        for name in (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKJP-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
            "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
            "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
        )
    )

    # macOS Japanese fonts.
    paths.extend(
        Path(name)
        for name in (
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
            "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc",
            "/Library/Fonts/NotoSansCJKjp-Regular.otf",
        )
    )
    return paths


def _font_has_required_glyphs(path: Path) -> bool:
    try:
        font = font_manager.get_font(str(path))
        charmap = font.get_charmap()
        return all(ord(char) in charmap for char in _REQUIRED_GLYPHS)
    except Exception:
        return False


def _existing_candidates(explicit_path: str | Path | None = None) -> Iterable[Path]:
    seen: set[str] = set()

    raw_paths: list[Path] = []
    if explicit_path:
        raw_paths.append(Path(explicit_path).expanduser())
    for key in _ENV_KEYS:
        value = os.environ.get(key)
        if value:
            raw_paths.append(Path(value).expanduser())
    raw_paths.extend(_common_font_paths())

    # Matplotlib's discovered fonts are also inspected, but only after explicit
    # and known OS paths.  This avoids relying on stale font-cache family names.
    family_rank = {name.lower(): rank for rank, name in enumerate(_PREFERRED_FAMILIES)}
    discovered = sorted(
        font_manager.fontManager.ttflist,
        key=lambda item: family_rank.get(item.name.lower(), len(family_rank) + 1),
    )
    raw_paths.extend(Path(item.fname) for item in discovered if item.name in _PREFERRED_FAMILIES)

    for path in raw_paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            yield resolved


@lru_cache(maxsize=8)
def resolve_japanese_font(explicit_path: str | None = None) -> tuple[str, Path]:
    """Return a Japanese-capable Matplotlib family and local font path.

    The function deliberately raises when no Japanese font is available rather
    than silently producing a garbled PDF.  Users can set
    STOCK_SCORING_JAPANESE_FONT to a local .ttf/.otf/.ttc path.
    """
    checked: list[str] = []
    for path in _existing_candidates(explicit_path):
        checked.append(str(path))
        if not _font_has_required_glyphs(path):
            continue
        # Register the concrete file even when Matplotlib's cache is stale.
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
        return family, path

    system_name = platform.system() or "Unknown OS"
    hint = (
        "日本語PDF用フォントを検出できませんでした。"
        "Windowsでは通常 Yu Gothic または Meiryo が利用できます。"
        "検出に失敗する場合は環境変数 STOCK_SCORING_JAPANESE_FONT に、"
        "ローカルPC上の .ttf/.otf/.ttc ファイルの絶対パスを設定してください。"
    )
    details = "\n候補として確認したパス:\n- " + "\n- ".join(checked[:30]) if checked else ""
    raise RuntimeError(f"{hint}\nOS: {system_name}{details}")


def setup_japanese_matplotlib(explicit_path: str | Path | None = None) -> tuple[str, Path]:
    """Apply a verified Japanese font to all Matplotlib PDF output."""
    family, path = resolve_japanese_font(str(explicit_path) if explicit_path else None)
    rcParams.update(
        {
            "font.family": family,
            "font.sans-serif": [family],
            "axes.unicode_minus": False,
            "font.size": 9,
            # Embed TrueType glyphs instead of Type 3 bitmap-like fonts.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.use14corefonts": False,
            "svg.fonttype": "none",
            "text.usetex": False,
        }
    )
    return family, path


def japanese_font_diagnostics(explicit_path: str | Path | None = None) -> dict[str, str]:
    family, path = setup_japanese_matplotlib(explicit_path)
    return {
        "family": family,
        "path": str(path),
        "environment_variable": "STOCK_SCORING_JAPANESE_FONT",
    }


def configured_font_path(config: dict, root: str | Path) -> Path | None:
    """Resolve the optional project-relative font path from model_config.py."""
    raw = config.get("reporting", {}).get("japanese_font_path")
    if raw is None or str(raw).strip() == "":
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def setup_japanese_matplotlib_from_config(config: dict, root: str | Path) -> tuple[str, Path]:
    return setup_japanese_matplotlib(configured_font_path(config, root))
