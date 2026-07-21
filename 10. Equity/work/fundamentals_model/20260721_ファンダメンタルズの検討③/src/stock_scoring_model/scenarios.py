from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .layer1_oof import generate_layer1_oof_subscores
from .layer2_factor_aggregation import aggregate_layer2_factor_scores
from .layer3_scope_selector import run_layer3_scopes
from .master import FactorMeta
from .preprocessing import build_factor_scores, percentile_rank
from .sector_grouping import apply_country_region_map, apply_sector_group_map


@dataclass
class ScenarioResult:
    stock_scores: pd.DataFrame
    factor_scores: pd.DataFrame  # Layer2 FactorScore long
    sub_scores: pd.DataFrame     # Layer1 SubScore long
    weight_history: pd.DataFrame
    model_selection: pd.DataFrame


def _assign_quintile(s: pd.Series, q: int) -> pd.Series:
    valid = s.notna()
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    if int(valid.sum()) >= q:
        ranks = s.loc[valid].rank(method="first")
        out.loc[valid] = pd.qcut(ranks, q=q, labels=range(1, q + 1)).astype(int)
    return out


def _score_rank(data: pd.DataFrame, prediction: pd.Series, config: dict[str, Any], rank_scope: str = "global") -> pd.Series:
    c = config["columns"]
    country_series = data[c["country"]] if c.get("country") in data.columns else pd.Series("ALL", index=data.index)
    tmp = pd.DataFrame({"date": data[c["date"]], "country": country_series, "prediction": prediction})
    if rank_scope == "country":
        return tmp.groupby(["date", "country"])["prediction"].transform(percentile_rank)
    return tmp.groupby("date")["prediction"].transform(percentile_rank)


def _stock_frame(
    data: pd.DataFrame,
    prediction: pd.Series,
    config: dict[str, Any],
    rank_scope: str = "global",
    extra_columns: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    c = config["columns"]
    out = pd.DataFrame({
        "Date": data[c["date"]],
        "ISIN": data[c["isin"]],
        "Country": data[c["country"]] if c.get("country") in data.columns else pd.Series(pd.NA, index=data.index),
        "Sector": data[c["sector"]] if c.get("sector") in data.columns else pd.Series(pd.NA, index=data.index),
        "Currency": data[c["currency"]],
        "MarketCap": data[c["market_cap"]],
        "Prediction": prediction,
        "NextMonthReturn": data["NextMonthReturn"],
    })
    if extra_columns:
        for name, series in extra_columns.items():
            out[name] = series
    out["RankScope"] = rank_scope
    out["TotalScore"] = _score_rank(data, prediction, config, rank_scope)
    qn = int(config["evaluation"].get("quintiles", 5))
    group_cols = ["Date", "Country"] if rank_scope == "country" else ["Date"]
    qseries = out.groupby(group_cols, group_keys=False)["TotalScore"].apply(lambda s: _assign_quintile(s, qn))
    if isinstance(qseries.index, pd.MultiIndex):
        qseries = qseries.reset_index(level=list(range(qseries.index.nlevels - 1)), drop=True)
    out["Quintile"] = qseries.reindex(out.index)
    return out


def _layer1_long(data: pd.DataFrame, subscores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    c = config["columns"]
    if subscores.empty:
        return pd.DataFrame(columns=["Date", "ISIN", "FactorCode", "SubScore"])
    tmp = pd.concat([data[[c["date"], c["isin"]]].reset_index(drop=True), subscores.reset_index(drop=True)], axis=1)
    return tmp.melt(id_vars=[c["date"], c["isin"]], var_name="FactorCode", value_name="SubScore").rename(columns={c["date"]: "Date", c["isin"]: "ISIN"})


def _layer2_long(data: pd.DataFrame, factor_scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    c = config["columns"]
    if factor_scores.empty:
        return pd.DataFrame(columns=["Date", "ISIN", "FactorGroup", "FactorScore"])
    tmp = pd.concat([data[[c["date"], c["isin"]]].reset_index(drop=True), factor_scores.reset_index(drop=True)], axis=1)
    return tmp.melt(id_vars=[c["date"], c["isin"]], var_name="FactorGroup", value_name="FactorScore").rename(columns={c["date"]: "Date", c["isin"]: "ISIN"})


def _simple_group_scores(scores: pd.DataFrame, metas: dict[str, FactorMeta]) -> pd.DataFrame:
    groups: dict[str, list[str]] = {}
    for code, meta in metas.items():
        if code in scores.columns:
            groups.setdefault(meta.group, []).append(code)
    return pd.DataFrame({group: scores[codes].mean(axis=1, skipna=True) for group, codes in groups.items()}, index=scores.index)


def _correlation_adjusted_groups(
    data: pd.DataFrame,
    scores: pd.DataFrame,
    metas: dict[str, FactorMeta],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # S05比較用。過去ICと当月相関から簡易相関調整ウェイトを作る。
    c = config["columns"]
    dates = sorted(pd.to_datetime(data[c["date"]].dropna().unique()))
    lookback = int(config["layer2"].get("ic_lookback_periods", 36))
    minp = int(config["layer2"].get("ic_minimum_periods", 12))
    groups: dict[str, list[str]] = {}
    for code, meta in metas.items():
        if code in scores:
            groups.setdefault(meta.group, []).append(code)
    ic_rows = []
    for date, idx in data.groupby(c["date"]).groups.items():
        y = data.loc[idx, "NextMonthReturn"]
        for code in scores.columns:
            mask = scores.loc[idx, code].notna() & y.notna()
            ic = spearmanr(scores.loc[idx[mask], code], y.loc[idx[mask]]).statistic if mask.sum() >= 8 else np.nan
            ic_rows.append({"Date": date, "FactorCode": code, "RankIC": ic})
    ic = pd.DataFrame(ic_rows)
    result = pd.DataFrame(index=data.index)
    wh = []
    for group, codes in groups.items():
        value = pd.Series(np.nan, index=data.index)
        for date in dates:
            idx = data.index[data[c["date"]].eq(date)]
            past = [d for d in dates if d < date][-lookback:]
            hist = ic[(ic["Date"].isin(past)) & ic["FactorCode"].isin(codes)]
            mu = hist.groupby("FactorCode")["RankIC"].mean().reindex(codes).clip(lower=0).fillna(0)
            count = hist.groupby("FactorCode")["RankIC"].count().reindex(codes).fillna(0)
            if (count < minp).any() or mu.sum() <= 0:
                w = np.ones(len(codes)) / len(codes)
                reason = "fallback_equal_weight"
            else:
                corr = scores.loc[idx, codes].corr().fillna(0).to_numpy(float)
                corr = 0.8 * corr + 0.2 * np.eye(len(codes))
                raw = np.clip(np.linalg.pinv(corr) @ mu.to_numpy(float), 0, None)
                w = raw / raw.sum() if raw.sum() > 0 else np.ones(len(codes)) / len(codes)
                reason = "correlation_adjusted_ic"
            arr = scores.loc[idx, codes].to_numpy(float)
            valid = np.isfinite(arr)
            num = np.nansum(arr * w, axis=1)
            den = np.sum(valid * w, axis=1)
            value.loc[idx] = np.divide(num, den, out=np.full(len(idx), np.nan), where=den > 0)
            for code, weight in zip(codes, w):
                wh.append({"Date": date, "Factor_Group": group, "FactorCode": code, "Weight": weight, "Reason": reason})
        result[group] = value
    return result, pd.DataFrame(wh)


def _finalize(
    name: str,
    data: pd.DataFrame,
    prediction: pd.Series,
    config: dict[str, Any],
    subscores: pd.DataFrame | None = None,
    factor_scores: pd.DataFrame | None = None,
    rank_scope: str = "global",
    weights: pd.DataFrame | None = None,
    selection: pd.DataFrame | None = None,
    extra: dict[str, pd.Series] | None = None,
) -> ScenarioResult:
    stock = _stock_frame(data, prediction, config, rank_scope=rank_scope, extra_columns=extra)
    stock.insert(0, "Scenario", name)
    return ScenarioResult(
        stock_scores=stock,
        factor_scores=_layer2_long(data, factor_scores if factor_scores is not None else pd.DataFrame(index=data.index), config),
        sub_scores=_layer1_long(data, subscores if subscores is not None else pd.DataFrame(index=data.index), config),
        weight_history=weights if weights is not None else pd.DataFrame(),
        model_selection=selection if selection is not None else pd.DataFrame(),
    )


def build_scenarios(
    data: pd.DataFrame,
    config: dict[str, Any],
    raw_metas: dict[str, FactorMeta],
    all_metas: dict[str, FactorMeta],
    group_methods: dict[str, str],
    country_region_map: pd.DataFrame,
    sector_group_map: pd.DataFrame,
    interaction_map: pd.DataFrame,
) -> tuple[dict[str, ScenarioResult], dict[str, dict[str, pd.DataFrame | pd.Series]], dict[str, pd.DataFrame]]:
    enabled = config.get("scenarios", {})
    results: dict[str, ScenarioResult] = {}
    raw_scores = build_factor_scores(data, config, raw_metas, winsorize=False, neutralize=False, rank_transform="uniform_0_1")
    win_scores = build_factor_scores(data, config, raw_metas, winsorize=True, neutralize=False, rank_transform="uniform_0_1")
    neu_scores = build_factor_scores(data, config, raw_metas, winsorize=True, neutralize=True, rank_transform="uniform_0_1")
    raw_cols = list(raw_scores.columns)

    if enabled.get("S00_Current_Direct_EW", True):
        pred = raw_scores[raw_cols].fillna(0.5).mean(axis=1)
        results["S00_Current_Direct_EW"] = _finalize("S00_Current_Direct_EW", data, pred, config, subscores=raw_scores, factor_scores=_simple_group_scores(raw_scores, raw_metas))
    if enabled.get("S01_Missing_Adjusted_EW", True):
        pred = raw_scores[raw_cols].mean(axis=1, skipna=True)
        results["S01_Missing_Adjusted_EW"] = _finalize("S01_Missing_Adjusted_EW", data, pred, config, subscores=raw_scores, factor_scores=_simple_group_scores(raw_scores, raw_metas))
    if enabled.get("S02_Winsorized_Direct_EW", True):
        pred = win_scores.mean(axis=1, skipna=True)
        results["S02_Winsorized_Direct_EW"] = _finalize("S02_Winsorized_Direct_EW", data, pred, config, subscores=win_scores, factor_scores=_simple_group_scores(win_scores, raw_metas))
    if enabled.get("S03_Neutralized_Direct_EW", True):
        pred = neu_scores.mean(axis=1, skipna=True)
        results["S03_Neutralized_Direct_EW"] = _finalize("S03_Neutralized_Direct_EW", data, pred, config, subscores=neu_scores, factor_scores=_simple_group_scores(neu_scores, raw_metas))
    if enabled.get("S04_Hierarchical_Equal_Weight", True):
        groups = _simple_group_scores(win_scores, raw_metas)
        pred = groups.mean(axis=1, skipna=True)
        results["S04_Hierarchical_Equal_Weight"] = _finalize("S04_Hierarchical_Equal_Weight", data, pred, config, subscores=win_scores, factor_scores=groups)
    if enabled.get("S05_Correlation_Adjusted_IC", True):
        groups, weights = _correlation_adjusted_groups(data, win_scores, raw_metas, config)
        pred = groups.mean(axis=1, skipna=True)
        results["S05_Correlation_Adjusted_IC"] = _finalize("S05_Correlation_Adjusted_IC", data, pred, config, subscores=win_scores, factor_scores=groups, weights=weights)

    # 第1層：Raw + 派生FAをグローバルで学習。説明変数はWinsorize後Gaussian Rank。
    layer1_input = build_factor_scores(data, config, all_metas, winsorize=True, neutralize=False, rank_transform="gaussian")
    (
        layer1_subscores,
        layer1_selection,
        layer1_coefficients,
        layer1_fit_history,
    ) = generate_layer1_oof_subscores(data, layer1_input, all_metas, config)
    layer2_scores, layer2_weights = aggregate_layer2_factor_scores(data, layer1_subscores, all_metas, group_methods, config)
    layer2_prediction = layer2_scores.mean(axis=1, skipna=True)

    if enabled.get("S06_Selected_Factor_Models", True):
        results["S06_Selected_Factor_Models"] = _finalize(
            "S06_Selected_Factor_Models", data, layer2_prediction, config,
            subscores=layer1_subscores, factor_scores=layer2_scores,
            weights=layer2_weights, selection=layer1_selection,
        )

    region = apply_country_region_map(data, config["columns"]["country"], country_region_map)
    sector_group = apply_sector_group_map(data, config["columns"]["sector"], sector_group_map)

    # 第3層の推定範囲比較は、既定のRidge設定で作成する。
    layer3 = run_layer3_scopes(data, layer2_scores, region, sector_group, interaction_map, config)
    primary_scope = str(config["layer3"].get("primary_scope", "country_independent"))
    if primary_scope not in layer3:
        primary_scope = next(iter(layer3))

    # S07はOLS/Ridgeを同じ線形基底・同じOOS開始条件で並列比較する。
    s07_variant_outputs: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
    variant_specs = config["layer3"].get("s07_variants", {})
    rank_scope = str(config["layer3"].get("final_score_rank_scope", "country"))
    for variant_name, spec in variant_specs.items():
        if not bool(spec.get("enabled", True)) or not enabled.get(variant_name, True):
            continue
        variant_config = deepcopy(config)
        variant_config["layer3"]["estimator"] = str(spec.get("estimator", "ridge"))
        # S07推定方式比較ではprimary_scopeのみ実行し、地域比較の重複計算を避ける。
        variant_config["layer3"]["comparison_scopes"] = [primary_scope]
        basis = list(spec.get("nonlinear_basis", ["linear"]))
        variant_config["layer3"]["nonlinear_basis"] = basis
        variant_config["layer3"]["include_nonlinear_basis"] = basis != ["linear"]
        scoped = run_layer3_scopes(data, layer2_scores, region, sector_group, interaction_map, variant_config)
        variant_scope = primary_scope if primary_scope in scoped else next(iter(scoped))
        payload = scoped[variant_scope]
        payload = dict(payload)
        payload["VariantName"] = pd.Series(variant_name, index=data.index)
        payload["Estimator"] = pd.Series(str(spec.get("estimator", "ridge")), index=data.index)
        payload["Basis"] = pd.Series(",".join(basis), index=data.index)
        s07_variant_outputs[variant_name] = payload
        results[variant_name] = _finalize(
            variant_name, data, payload["Prediction"], config,
            subscores=layer1_subscores, factor_scores=layer2_scores,
            rank_scope=rank_scope, weights=layer2_weights, selection=layer1_selection,
            extra={
                "Layer3Scope": pd.Series(variant_scope, index=data.index),
                "Layer3Estimator": pd.Series(str(spec.get("estimator", "ridge")), index=data.index),
                "Layer3Basis": pd.Series(",".join(basis), index=data.index),
                "Region": region,
                "SectorGroup": sector_group,
            },
        )

    diagnostics = {
        "Layer1Selection": layer1_selection,
        "Layer1Coefficients": layer1_coefficients,
        "Layer1FitHistory": layer1_fit_history,
        "Layer1InputScores": layer1_input,
        "Layer1Subscores": layer1_subscores,
        "Layer2Weights": layer2_weights,
        "Layer2FactorScores": layer2_scores,
        "Region": pd.DataFrame({"Region": region}),
        "SectorGroup": pd.DataFrame({"SectorGroup": sector_group}),
        "S07Variants": s07_variant_outputs,
    }
    return results, layer3, diagnostics
