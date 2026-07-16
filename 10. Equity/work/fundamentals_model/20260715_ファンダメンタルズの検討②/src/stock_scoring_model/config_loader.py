from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "preprocessing": {
        "winsorize_lower": 0.01,
        "winsorize_upper": 0.99,
        "minimum_cross_section": 20,
        "rank_transform": "uniform_0_1",
        "gaussian_clip": 3.0,
    },
    "model": {
        "training_window_periods": 36,
        "minimum_train_periods": 18,
        "ic_lookback_periods": 36,
        "ic_minimum_periods": 12,
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "one_se_rule": True,
    },
}


def _deep_update(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("user_model_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Configを読み込めません: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise ValueError("ConfigファイルにCONFIG辞書がありません。")
    cfg = _deep_update(deepcopy(DEFAULTS), deepcopy(module.CONFIG))
    root = path.parent.parent
    return cfg, root


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p
