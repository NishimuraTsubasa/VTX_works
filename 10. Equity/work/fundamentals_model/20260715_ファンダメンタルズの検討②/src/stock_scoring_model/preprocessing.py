from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

from .factor_master import factor_lookup
from .utils import rank_to_unit_interval

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    panel: pd.DataFrame
    processed_factors: list[str]
    quality: pd.DataFrame


def align_forward_target(stocks: pd.DataFrame, config: dict) -> pd.DataFrame:
    cols = config["columns"]
    date_col, isin_col, ret_col = cols["date"], cols["isin"], cols["stock_return"]
    if ret_col not in stocks:
        raise ValueError(f"Stock return column '{ret_col}' is missing.")
    out = stocks.sort_values([isin_col, date_col]).copy()
    alignment = config["target"]["stock_return_alignment"]
    horizon = int(config["target"].get("stock_horizon_periods", 1))
    if alignment == "already_forward":
        out["forward_return"] = pd.to_numeric(out[ret_col], errors="coerce")
    elif alignment == "contemporaneous_to_forward":
        out["forward_return"] = out.groupby(isin_col)[ret_col].shift(-horizon)
    else:
        raise ValueError(f"Unknown stock_return_alignment: {alignment}")
    return out


def _transform_raw(s: pd.Series, method: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if method == "none":
        return x
    if method == "log":
        return np.log(x.where(x > 0))
    if method == "log1p":
        return np.log1p(x.where(x > -1))
    if method == "inverse":
        return 1.0 / x.where(x.abs() > 1e-12)
    if method == "signed_log":
        return np.sign(x) * np.log1p(x.abs())
    raise ValueError(f"Unknown factor transform: {method}")


def _winsorize_quantile(s: pd.Series, lower: float, upper: float, min_obs: int) -> tuple[pd.Series, float, float, int]:
    valid = s.dropna()
    if len(valid) < min_obs:
        return s.copy(), np.nan, np.nan, 0
    lo, hi = valid.quantile([lower, upper])
    clipped = s.clip(lo, hi)
    count = int(((s < lo) | (s > hi)).sum())
    return clipped, float(lo), float(hi), count


def _winsorize_mad(s: pd.Series, threshold: float = 3.0) -> tuple[pd.Series, float, float, int]:
    valid = s.dropna()
    if len(valid) < 10:
        return s.copy(), np.nan, np.nan, 0
    med = float(valid.median())
    mad = float((valid - med).abs().median())
    if not np.isfinite(mad) or mad <= 1e-12:
        return s.copy(), np.nan, np.nan, 0
    scale = 1.4826 * mad
    lo, hi = med - threshold * scale, med + threshold * scale
    clipped = s.clip(lo, hi)
    count = int(((s < lo) | (s > hi)).sum())
    return clipped, lo, hi, count


def _winsorize_by_setting(s: pd.Series, setting: str, config: dict) -> tuple[pd.Series, float, float, int, str]:
    default = config["preprocessing"]["winsorize"]
    min_obs = int(default.get("minimum_observations", 20))
    setting = str(setting or "default").lower()
    if setting == "default":
        if not default.get("enabled", True):
            return s.copy(), np.nan, np.nan, 0, "none"
        lower = float(default["lower_quantile"])
        upper = float(default["upper_quantile"])
        out = _winsorize_quantile(s, lower, upper, min_obs)
        return (*out, f"{100*lower:g}_{100*upper:g}")
    if setting == "none":
        return s.copy(), np.nan, np.nan, 0, "none"
    if setting == "1_99":
        return (*_winsorize_quantile(s, 0.01, 0.99, min_obs), "1_99")
    if setting == "2.5_97.5":
        return (*_winsorize_quantile(s, 0.025, 0.975, min_obs), "2.5_97.5")
    if setting == "mad_3":
        return (*_winsorize_mad(s, 3.0), "mad_3")
    raise ValueError(f"Unknown Winsorize setting: {setting}")


def _neutralize(group: pd.DataFrame, factor: str, config: dict, enabled: bool = True) -> pd.Series:
    ncfg = config["preprocessing"]["neutralization"]
    if not enabled or not ncfg.get("enabled", True):
        return pd.to_numeric(group[factor], errors="coerce")

    cols = config["columns"]
    categorical = []
    for token in ncfg.get("categorical", []):
        col = cols.get(token, token)
        if col in group.columns:
            categorical.append(col)

    design_parts: list[pd.DataFrame] = []
    if categorical:
        design_parts.append(pd.get_dummies(group[categorical], dummy_na=True, drop_first=True, dtype=float))

    for token in ncfg.get("numeric", []):
        if token == "log_market_cap":
            mcap_col = cols["market_cap"]
            if mcap_col in group.columns:
                values = pd.to_numeric(group[mcap_col], errors="coerce")
                design_parts.append(pd.DataFrame({"log_market_cap": np.log(values.where(values > 0))}, index=group.index))
        elif token in group.columns:
            design_parts.append(pd.DataFrame({token: pd.to_numeric(group[token], errors="coerce")}, index=group.index))

    y = pd.to_numeric(group[factor], errors="coerce")
    if not design_parts:
        return y
    X = pd.concat(design_parts, axis=1)
    valid = y.notna() & X.notna().all(axis=1)
    if valid.sum() < int(ncfg.get("minimum_observations", 30)) or X.loc[valid].shape[1] == 0:
        return y - y.mean()

    model = Ridge(alpha=float(ncfg.get("ridge_alpha", 1e-6)), fit_intercept=True)
    model.fit(X.loc[valid], y.loc[valid])
    residual = pd.Series(np.nan, index=group.index, dtype=float)
    residual.loc[valid] = y.loc[valid] - model.predict(X.loc[valid])
    return residual


def _rank_transform(s: pd.Series, method: str, gaussian_clip: float) -> pd.Series:
    u = rank_to_unit_interval(s)
    if method == "uniform_0_1":
        return u
    if method == "uniform_minus1_1":
        return 2.0 * u - 1.0
    if method == "gaussian":
        out = pd.Series(np.nan, index=s.index, dtype=float)
        valid = u.notna()
        out.loc[valid] = norm.ppf(u.loc[valid].clip(1e-6, 1 - 1e-6))
        return out.clip(-gaussian_clip, gaussian_clip)
    raise ValueError(f"Unknown rank_transform: {method}")


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if not np.isfinite(std) or std <= 1e-12:
        return s * 0.0
    return (s - s.mean()) / std


def preprocess_panel(
    stocks: pd.DataFrame,
    factor_columns: list[str],
    config: dict,
    factor_master: pd.DataFrame | None = None,
) -> PreprocessResult:
    cols = config["columns"]
    date_col = cols["date"]
    out = align_forward_target(stocks, config)
    quality_rows: list[dict] = []
    processed: list[str] = []
    lookup = factor_lookup(factor_master) if factor_master is not None and not factor_master.empty else {}

    for factor in factor_columns:
        settings = lookup.get(factor, {})
        direction = int(settings.get("Direction", 1))
        transform_method = str(settings.get("Transform", "none")).lower()
        winsor_setting = str(settings.get("Winsorize", "default")).lower()
        neutralize_enabled = bool(int(settings.get("Neutralize", 1)))
        rank_enabled = bool(int(settings.get("Rank_Normalize", 1)))
        min_coverage = float(settings.get("Min_Coverage", 0.0))
        group_name = settings.get("Factor_Group", "")

        raw_col = f"{factor}__transformed"
        win_col = f"{factor}__win"
        neu_col = f"{factor}__neutral"
        z_col = f"{factor}__z"
        out[raw_col] = _transform_raw(out[factor], transform_method)
        out[win_col] = np.nan
        out[neu_col] = np.nan
        out[z_col] = np.nan

        for date, idx in out.groupby(date_col).groups.items():
            g = out.loc[idx]
            raw = pd.to_numeric(g[raw_col], errors="coerce")
            coverage = float(raw.notna().mean())
            win, lo, hi, clip_count, resolved_winsor = _winsorize_by_setting(raw, winsor_setting, config)
            out.loc[idx, win_col] = win
            temp = g.copy()
            temp[win_col] = win
            neutral = _neutralize(temp, win_col, config, neutralize_enabled)
            out.loc[idx, neu_col] = neutral
            if coverage < min_coverage:
                transformed = pd.Series(np.nan, index=neutral.index, dtype=float)
                status = "below_min_coverage"
            elif rank_enabled:
                transformed = _rank_transform(
                    neutral,
                    config["preprocessing"]["rank_transform"],
                    float(config["preprocessing"].get("gaussian_clip", 3.0)),
                )
                status = "ok"
            else:
                transformed = _zscore(neutral)
                status = "ok_no_rank"
            transformed = transformed * direction
            out.loc[idx, z_col] = transformed
            quality_rows.append({
                "date": date,
                "factor": factor,
                "factor_group": group_name,
                "factor_name_jp": settings.get("Factor_Name_JP", factor),
                "observations": int(raw.notna().sum()),
                "coverage": coverage,
                "minimum_coverage": min_coverage,
                "status": status,
                "transform": transform_method,
                "direction": direction,
                "winsor_method": resolved_winsor,
                "winsor_lower": lo,
                "winsor_upper": hi,
                "winsorized_count": clip_count,
                "winsorized_rate": float(clip_count / max(raw.notna().sum(), 1)),
                "neutralized": neutralize_enabled,
                "rank_normalized": rank_enabled,
                "processed_mean": float(transformed.mean()) if transformed.notna().any() else np.nan,
                "processed_std": float(transformed.std()) if transformed.notna().sum() > 1 else np.nan,
            })

        missing_indicator = f"{factor}__missing"
        if config["preprocessing"].get("add_missing_indicators", False):
            out[missing_indicator] = out[z_col].isna().astype(int)
        out[z_col] = out[z_col].fillna(float(config["preprocessing"]["fill_missing_value"]))
        processed.append(z_col)

    return PreprocessResult(out, processed, pd.DataFrame(quality_rows))
