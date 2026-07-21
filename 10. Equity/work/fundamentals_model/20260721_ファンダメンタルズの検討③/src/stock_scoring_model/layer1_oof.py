from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .layer1_model_selection import select_candidate_model
from .layer1_single_factor import fit_single_factor
from .master import FactorMeta


def generate_layer1_oof_subscores(
    data: pd.DataFrame,
    factor_scores: pd.DataFrame,
    metas: dict[str, FactorMeta],
    config: dict[str, Any],
    target_col: str = "NextMonthReturn",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """グローバルユニバースで各FAのWalk-forward OOF SubScoreを生成する。"""
    c = config["columns"]
    cfg = config["layer1"]
    dates = sorted(pd.to_datetime(data[c["date"]].dropna().unique()))
    window = int(cfg.get("training_window_periods", 36))
    min_train = int(cfg.get("minimum_train_periods", 18))
    val_periods = int(cfg.get("validation_periods", 6))
    min_fit_obs = int(cfg.get("minimum_fit_observations", 200))
    min_val_obs = int(cfg.get("minimum_validation_observations", 100))
    predictions = pd.DataFrame(index=data.index)
    selection_rows: list[pd.DataFrame] = []

    for code in [k for k in metas if k in factor_scores.columns]:
        output = pd.Series(np.nan, index=data.index, dtype=float)
        for pos, date in enumerate(dates):
            train_dates = dates[max(0, pos - window):pos]
            if len(train_dates) < min_train or len(train_dates) <= val_periods:
                continue
            fit_dates = train_dates[:-val_periods]
            valid_dates = train_dates[-val_periods:]
            fit_idx = data.index[data[c["date"]].isin(fit_dates)]
            valid_idx = data.index[data[c["date"]].isin(valid_dates)]
            test_idx = data.index[data[c["date"]].eq(date)]
            fit_mask = factor_scores.loc[fit_idx, code].notna() & data.loc[fit_idx, target_col].notna()
            valid_mask = factor_scores.loc[valid_idx, code].notna() & data.loc[valid_idx, target_col].notna()
            if fit_mask.sum() < min_fit_obs or valid_mask.sum() < min_val_obs:
                continue
            selected, candidates = select_candidate_model(
                factor_scores.loc[fit_idx[fit_mask], code].to_numpy(float),
                data.loc[fit_idx[fit_mask], target_col].to_numpy(float),
                factor_scores.loc[valid_idx[valid_mask], code].to_numpy(float),
                data.loc[valid_idx[valid_mask], target_col].to_numpy(float),
                data.loc[valid_idx[valid_mask], c["date"]],
                config,
            )
            full_idx = data.index[data[c["date"]].isin(train_dates)]
            full_mask = factor_scores.loc[full_idx, code].notna() & data.loc[full_idx, target_col].notna()
            test_mask = factor_scores.loc[test_idx, code].notna()
            model = fit_single_factor(
                factor_scores.loc[full_idx[full_mask], code].to_numpy(float),
                data.loc[full_idx[full_mask], target_col].to_numpy(float),
                selected,
                knot=float(cfg.get("piecewise_knot", 0.0)),
                ridge_alpha=float(cfg.get("ridge_alpha", 1e-8)),
            )
            if test_mask.any():
                output.loc[test_idx[test_mask]] = model.predict(
                    factor_scores.loc[test_idx[test_mask], code].to_numpy(float)
                )
            candidates.insert(0, "Date", date)
            candidates.insert(1, "FactorCode", code)
            selection_rows.append(candidates)
        predictions[code] = output
    selection = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    return predictions, selection
