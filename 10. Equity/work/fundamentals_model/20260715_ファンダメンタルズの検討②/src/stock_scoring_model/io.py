from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config_loader import resolve_path


def read_inputs(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data_path = resolve_path(root, config["data"]["factors_file"])
    master_path = resolve_path(root, config["data"]["factor_master_file"])
    data = pd.read_excel(data_path, sheet_name=config["data"].get("factors_sheet", "data"))
    sheets = pd.read_excel(master_path, sheet_name=None)
    return data, sheets


def ensure_output_dirs(config: dict[str, Any], root: Path) -> dict[str, Path]:
    out = resolve_path(root, config["outputs"].get("output_dir", "outputs"))
    paths = {
        "root": out,
        "diagnostics": out / "diagnostics",
        "patterns": out / "stock_score_patterns",
        "history": out / "history",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
