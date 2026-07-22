from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .layer2_factor_return_weighting import aggregate_raw_factor_scores
from .layer3_scope_selector import run_layer3_scopes
from .master import FactorMeta
from .preprocessing import build_factor_scores, percentile_rank
from .sector_grouping import apply_country_region_map, apply_sector_group_map


@dataclass
class ScenarioResult:
    stock_scores: pd.DataFrame
    factor_scores: pd.DataFrame
    direct_factor_scores: pd.DataFrame
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
    country = data[c["country"]] if c.get("country") in data.columns else pd.Series("ALL", index=data.index)
    temp = pd.DataFrame({"Date": data[c["date"]], "Country": country, "Prediction": prediction})
    if rank_scope == "country":
        return temp.groupby(["Date", "Country"])["Prediction"].transform(percentile_rank)
    return temp.groupby("Date")["Prediction"].transform(percentile_rank)


def _stock_frame(
    data: pd.DataFrame,
    prediction: pd.Series,
    config: dict[str, Any],
    *,
    rank_scope: str = "global",
    extra_columns: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    c = config["columns"]
    out = pd.DataFrame({
        "Date": pd.to_datetime(data[c["date"]]),
        "ISIN": data[c["isin"]].astype(str),
        "Country": data[c["country"]],
        "Sector": data[c["sector"]],
        "Currency": data[c["currency"]],
        "MarketCap": data[c["market_cap"]],
        "Prediction": pd.to_numeric(prediction, errors="coerce"),
        "NextMonthReturn": data["NextMonthReturn"],
    })
    if extra_columns:
        for name, series in extra_columns.items():
            out[name] = series
    out["RankScope"] = rank_scope
    out["TotalScore"] = _score_rank(data, out["Prediction"], config, rank_scope)
    qn = int(config["evaluation"].get("quintiles", 5))
    group_cols = ["Date", "Country"] if rank_scope == "country" else ["Date"]
    qseries = out.groupby(group_cols, group_keys=False)["TotalScore"].apply(lambda s: _assign_quintile(s, qn))
    if isinstance(qseries.index, pd.MultiIndex):
        qseries = qseries.reset_index(level=list(range(qseries.index.nlevels - 1)), drop=True)
    out["Quintile"] = qseries.reindex(out.index)
    return out


def _score_long(
    data: pd.DataFrame,
    scores: pd.DataFrame,
    config: dict[str, Any],
    *,
    name_col: str,
    value_col: str,
) -> pd.DataFrame:
    c = config["columns"]
    if scores is None or scores.empty:
        return pd.DataFrame(columns=["Date", "ISIN", name_col, value_col])
    temp = pd.concat(
        [data[[c["date"], c["isin"]]].reset_index(drop=True), scores.reset_index(drop=True)],
        axis=1,
    )
    return temp.melt(
        id_vars=[c["date"], c["isin"]],
        var_name=name_col,
        value_name=value_col,
    ).rename(columns={c["date"]: "Date", c["isin"]: "ISIN"})


def _simple_group_scores(scores: pd.DataFrame, metas: dict[str, FactorMeta]) -> pd.DataFrame:
    groups: dict[str, list[str]] = {}
    for code, meta in metas.items():
        if code in scores.columns:
            groups.setdefault(meta.group, []).append(code)
    return pd.DataFrame(
        {group: scores[codes].mean(axis=1, skipna=True) for group, codes in groups.items()},
        index=scores.index,
    )


def _hierarchical_factor_count_prediction(scores: pd.DataFrame, metas: dict[str, FactorMeta]) -> pd.Series:
    """階層表示しつつ、各FAの実効ウェイトを等しくしてDirect EWを再現する。"""
    available_codes = [code for code in metas if code in scores.columns]
    return scores[available_codes].mean(axis=1, skipna=True)


def _finalize(
    name: str,
    data: pd.DataFrame,
    prediction: pd.Series,
    config: dict[str, Any],
    *,
    raw_factor_scores: pd.DataFrame | None = None,
    factor_scores: pd.DataFrame | None = None,
    rank_scope: str = "global",
    weights: pd.DataFrame | None = None,
    extra: dict[str, pd.Series] | None = None,
) -> ScenarioResult:
    stock = _stock_frame(data, prediction, config, rank_scope=rank_scope, extra_columns=extra)
    stock.insert(0, "Scenario", name)
    return ScenarioResult(
        stock_scores=stock,
        factor_scores=_score_long(
            data,
            factor_scores if factor_scores is not None else pd.DataFrame(index=data.index),
            config,
            name_col="FactorGroup",
            value_col="FactorScore",
        ),
        direct_factor_scores=_score_long(
            data,
            raw_factor_scores if raw_factor_scores is not None else pd.DataFrame(index=data.index),
            config,
            name_col="FactorCode",
            value_col="RawFactorScore",
        ),
        weight_history=weights if weights is not None else pd.DataFrame(),
        model_selection=pd.DataFrame(),
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
) -> tuple[dict[str, ScenarioResult], dict[str, dict[str, pd.DataFrame | pd.Series]], dict[str, Any]]:
    del group_methods  # v0.13ではConfigのFactorReturn相関ウェイトを本線とする。
    enabled = config.get("scenarios", {})
    results: dict[str, ScenarioResult] = {}

    layer2_cfg = config.get("layer2", {})
    use_derived = str(layer2_cfg.get("factor_universe", "raw_only")) == "raw_and_derived"
    metas = all_metas if use_derived else raw_metas
    score_transform = str(layer2_cfg.get("raw_score_transform", "centered_percentile"))
    use_neutralized = bool(layer2_cfg.get("use_neutralized_scores", True))
    direct_scores = build_factor_scores(
        data,
        config,
        metas,
        winsorize=bool(layer2_cfg.get("winsorize", True)),
        neutralize=use_neutralized,
        rank_transform=score_transform,
    )
    group_equal_scores = _simple_group_scores(direct_scores, metas)

    if enabled.get("N00_Direct_RawScore_EW", True):
        pred = direct_scores.mean(axis=1, skipna=True)
        results["N00_Direct_RawScore_EW"] = _finalize(
            "N00_Direct_RawScore_EW",
            data,
            pred,
            config,
            raw_factor_scores=direct_scores,
            factor_scores=group_equal_scores,
        )

    if enabled.get("N01_Hierarchical_FactorCount_EW", True):
        pred = _hierarchical_factor_count_prediction(direct_scores, metas)
        results["N01_Hierarchical_FactorCount_EW"] = _finalize(
            "N01_Hierarchical_FactorCount_EW",
            data,
            pred,
            config,
            raw_factor_scores=direct_scores,
            factor_scores=group_equal_scores,
        )

    if enabled.get("N02_Hierarchical_Group_EW", True):
        pred = group_equal_scores.mean(axis=1, skipna=True)
        results["N02_Hierarchical_Group_EW"] = _finalize(
            "N02_Hierarchical_Group_EW",
            data,
            pred,
            config,
            raw_factor_scores=direct_scores,
            factor_scores=group_equal_scores,
        )

    corr_only = aggregate_raw_factor_scores(data, direct_scores, metas, config, equal_weight_blend=0.0)
    if enabled.get("N03_FactorReturn_Correlation", True):
        pred = corr_only.factor_scores.mean(axis=1, skipna=True)
        results["N03_FactorReturn_Correlation"] = _finalize(
            "N03_FactorReturn_Correlation",
            data,
            pred,
            config,
            raw_factor_scores=direct_scores,
            factor_scores=corr_only.factor_scores,
            weights=corr_only.weight_history,
        )

    shrunk = aggregate_raw_factor_scores(data, direct_scores, metas, config)
    if enabled.get("N04_FactorReturn_Correlation_ShrunkEW", True):
        pred = shrunk.factor_scores.mean(axis=1, skipna=True)
        results["N04_FactorReturn_Correlation_ShrunkEW"] = _finalize(
            "N04_FactorReturn_Correlation_ShrunkEW",
            data,
            pred,
            config,
            raw_factor_scores=direct_scores,
            factor_scores=shrunk.factor_scores,
            weights=shrunk.weight_history,
        )

    region = apply_country_region_map(data, config["columns"]["country"], country_region_map)
    sector_group = apply_sector_group_map(data, config["columns"]["sector"], sector_group_map)
    primary_scope = str(config["layer3"].get("primary_scope", "country_independent"))
    rank_scope = str(config["layer3"].get("final_score_rank_scope", "country"))
    variant_outputs: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}

    for variant_name, spec in config["layer3"].get("variants", {}).items():
        if not bool(spec.get("enabled", True)) or not enabled.get(variant_name, True):
            continue
        variant_config = deepcopy(config)
        variant_config["layer3"]["comparison_scopes"] = [primary_scope]
        variant_config["layer3"]["estimator"] = str(spec.get("estimator", "ridge"))
        variant_config["layer3"]["nonlinear_basis"] = list(spec.get("nonlinear_basis", ["linear"]))
        variant_config["layer3"]["include_nonlinear_basis"] = variant_config["layer3"]["nonlinear_basis"] != ["linear"]
        variant_config["layer3"]["include_sector_group_dummy"] = bool(spec.get("include_sector_group_dummy", False))
        variant_config["layer3"]["include_sector_factor_interactions"] = bool(spec.get("include_sector_factor_interactions", False))
        scoped = run_layer3_scopes(
            data,
            shrunk.factor_scores,
            region,
            sector_group,
            interaction_map,
            variant_config,
        )
        scope_name = primary_scope if primary_scope in scoped else next(iter(scoped))
        payload = dict(scoped[scope_name])
        payload["VariantName"] = pd.Series(variant_name, index=data.index)
        payload["Estimator"] = pd.Series(str(spec.get("estimator", "ridge")), index=data.index)
        variant_outputs[variant_name] = payload
        results[variant_name] = _finalize(
            variant_name,
            data,
            payload["Prediction"],
            config,
            raw_factor_scores=direct_scores,
            factor_scores=shrunk.factor_scores,
            rank_scope=rank_scope,
            weights=shrunk.weight_history,
            extra={
                "Layer3Scope": pd.Series(scope_name, index=data.index),
                "Layer3Estimator": pd.Series(str(spec.get("estimator", "ridge")), index=data.index),
                "Region": region,
                "SectorGroup": sector_group,
            },
        )

    # Scope比較は最も柔軟な有効variantの仕様で作る。
    scope_config = deepcopy(config)
    scope_variant = next(
        (spec for name, spec in reversed(list(config["layer3"].get("variants", {}).items())) if enabled.get(name, True) and spec.get("enabled", True)),
        {"estimator": "ridge", "include_sector_group_dummy": False, "include_sector_factor_interactions": False, "nonlinear_basis": ["linear"]},
    )
    scope_config["layer3"]["estimator"] = str(scope_variant.get("estimator", "ridge"))
    scope_config["layer3"]["include_sector_group_dummy"] = bool(scope_variant.get("include_sector_group_dummy", False))
    scope_config["layer3"]["include_sector_factor_interactions"] = bool(scope_variant.get("include_sector_factor_interactions", False))
    scope_config["layer3"]["nonlinear_basis"] = list(scope_variant.get("nonlinear_basis", ["linear"]))
    scope_config["layer3"]["include_nonlinear_basis"] = scope_config["layer3"]["nonlinear_basis"] != ["linear"]
    layer3 = run_layer3_scopes(data, shrunk.factor_scores, region, sector_group, interaction_map, scope_config)

    diagnostics: dict[str, Any] = {
        "DirectFactorScores": direct_scores,
        "Layer2FactorScores": shrunk.factor_scores,
        "Layer2Weights": shrunk.weight_history,
        "FactorReturnHistory": shrunk.factor_return_history,
        "FactorReturnCorrelations": shrunk.correlation_history,
        "N03Layer2Weights": corr_only.weight_history,
        "Region": pd.DataFrame({"Region": region}),
        "SectorGroup": pd.DataFrame({"SectorGroup": sector_group}),
        "Layer3Variants": variant_outputs,
    }
    return results, layer3, diagnostics
