from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from .master import FactorMeta


def winsorize_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    valid = x.dropna()
    if len(valid) < 5:
        return x
    lo, hi = valid.quantile([lower, upper])
    return x.clip(lo, hi)


def percentile_rank(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mask = x.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    n = int(mask.sum())
    if n:
        out.loc[mask] = (rankdata(x.loc[mask], method="average") - 0.5) / n
    return out


def gaussian_rank(s: pd.Series, clip: float = 3.0) -> pd.Series:
    u = percentile_rank(s)
    out = pd.Series(norm.ppf(u), index=s.index)
    return out.clip(-clip, clip)


def neutralize_cross_section(group: pd.DataFrame, value_col: str, config: dict[str, Any]) -> pd.Series:
    c = config["columns"]
    y = pd.to_numeric(group[value_col], errors="coerce")
    mask = y.notna() & pd.to_numeric(group[c["market_cap"]], errors="coerce").gt(0)
    if mask.sum() < int(config["preprocessing"].get("minimum_cross_section", 20)):
        return y - y.mean()
    tmp = group.loc[mask, [c["country"], c["sector"], c["market_cap"]]].copy()
    tmp["log_market_cap"] = np.log(pd.to_numeric(tmp[c["market_cap"]], errors="coerce"))
    mode = str(config.get("preprocessing", {}).get("neutralization_mode", "country_sector_and_size"))
    enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    if mode == "country_and_sector_and_size":
        cats = enc.fit_transform(tmp[[c["country"], c["sector"]]].astype(str))
    elif mode == "country_sector_only":
        tmp["country_sector"] = tmp[c["country"]].astype(str) + "||" + tmp[c["sector"]].astype(str)
        cats = enc.fit_transform(tmp[["country_sector"]])
        X = np.column_stack([np.ones(len(tmp)), cats])
        model = Ridge(alpha=float(config.get("preprocessing", {}).get("neutralization_ridge_alpha", 1e-6)), fit_intercept=False)
        model.fit(X, y.loc[mask].to_numpy())
        resid = y.loc[mask].to_numpy() - model.predict(X)
        out = pd.Series(np.nan, index=group.index, dtype=float)
        out.loc[mask] = resid
        return out
    else:
        tmp["country_sector"] = tmp[c["country"]].astype(str) + "||" + tmp[c["sector"]].astype(str)
        cats = enc.fit_transform(tmp[["country_sector"]])
    X = np.column_stack([np.ones(len(tmp)), cats, tmp["log_market_cap"].to_numpy()])
    model = Ridge(alpha=float(config.get("preprocessing", {}).get("neutralization_ridge_alpha", 1e-6)), fit_intercept=False)
    model.fit(X, y.loc[mask].to_numpy())
    resid = y.loc[mask].to_numpy() - model.predict(X)
    out = pd.Series(np.nan, index=group.index, dtype=float)
    out.loc[mask] = resid
    return out


def build_factor_scores(
    data: pd.DataFrame,
    config: dict[str, Any],
    metas: dict[str, FactorMeta],
    *,
    winsorize: bool,
    neutralize: bool,
    rank_transform: str,
) -> pd.DataFrame:
    c = config["columns"]
    # スコア行列は入力データと同じindexを持つFA列のみで構成する。
    # 日付・ISINを含めると、後段のIC計算が識別列をファクターとして誤認するため含めない。
    out = pd.DataFrame(index=data.index)
    lower = float(config["preprocessing"].get("winsorize_lower", 0.01))
    upper = float(config["preprocessing"].get("winsorize_upper", 0.99))
    clip = float(config["preprocessing"].get("gaussian_clip", 3.0))

    for code, meta in metas.items():
        if code not in data.columns:
            continue
        x = pd.to_numeric(data[code], errors="coerce")
        if winsorize:
            x = data.assign(__x=x).groupby(c["date"], group_keys=False)["__x"].apply(
                lambda s: winsorize_series(s, lower, upper)
            ).reset_index(level=0, drop=True)
        if neutralize:
            temp = data.copy()
            temp["__x"] = x
            x = temp.groupby(c["date"], group_keys=False).apply(
                lambda g: neutralize_cross_section(g, "__x", config), include_groups=False
            )
            x = x.reset_index(level=0, drop=True) if isinstance(x.index, pd.MultiIndex) else x
        temp = pd.DataFrame({c["date"]: data[c["date"]], "__x": x})
        if rank_transform == "gaussian":
            score = temp.groupby(c["date"], group_keys=False)["__x"].apply(lambda s: gaussian_rank(s, clip))
        else:
            score = temp.groupby(c["date"], group_keys=False)["__x"].apply(percentile_rank)
            if meta.direction == -1:
                score = 1.0 - score
        if rank_transform == "gaussian":
            score = score * meta.direction
        out[code] = score.reset_index(level=0, drop=True)
    return out


def make_diagnostic_score(
    frame: pd.DataFrame,
    value_col: str,
    date_col: str,
    direction: int,
    lower: float = 0.01,
    upper: float = 0.99,
    clip: float = 3.0,
) -> pd.Series:
    """Scope内・時点内でwinsorize -> gaussian rank -> direction調整。"""
    temp = frame[[date_col, value_col]].copy()
    temp["__w"] = temp.groupby(date_col)[value_col].transform(lambda s: winsorize_series(s, lower, upper))
    temp["__z"] = temp.groupby(date_col)["__w"].transform(lambda s: gaussian_rank(s, clip))
    return temp["__z"] * direction
