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
        "rank_transform": "gaussian",
        "gaussian_clip": 3.0,
        "neutralization_mode": "country_sector_and_size",
        "neutralization_ridge_alpha": 1e-6,
    },
    "layer1": {
        "candidate_models": ["linear", "piecewise", "quadratic"],
        "training_window_periods": 36,
        "minimum_train_periods": 18,
        "validation_periods": 6,
        "minimum_fit_observations": 200,
        "minimum_validation_observations": 100,
        "piecewise_knot": 0.0,
        "one_se_rule": True,
        "ridge_alpha": 1e-8,
    },
    "layer2": {
        "ic_lookback_periods": 36,
        "ic_minimum_periods": 12,
        "correlation_shrinkage": 0.20,
        "maximum_factor_weight": 0.60,
        "weight_smoothing": 0.50,
    },
    "layer3": {
        "primary_scope": "country_independent",
        "comparison_scopes": ["country_independent", "regional_pooling", "hierarchical_partial_pooling"],
        "training_mode": "rolling_pooled",
        "lookback_periods": 36,
        "minimum_train_periods": 18,
        "minimum_training_observations": 250,
        "include_nonlinear_basis": True,
        "nonlinear_basis": ["linear", "piecewise", "quadratic"],
        "piecewise_knot": 0.0,
        "include_sector_group_dummy": True,
        "include_sector_factor_interactions": True,
        "interaction_mode": "selected_interactions",
        "include_country_controls_in_regional": True,
        "demean_target_by_date": True,
        "ridge_alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
        "country_deviation_penalty_multiplier": 10.0,
        "country_intercept_penalty_multiplier": 2.0,
        "fallback_scope": "regional_pooling",
        "final_score_rank_scope": "country",
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
    return cfg, path.parent.parent


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
