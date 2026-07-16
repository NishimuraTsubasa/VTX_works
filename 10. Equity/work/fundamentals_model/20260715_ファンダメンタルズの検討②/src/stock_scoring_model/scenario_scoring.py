from __future__ import annotations

"""個別銘柄スコアリングの段階比較用データを作成する。

各シナリオは、同じ個別銘柄ユニバースに対して、ファクター値の加工・
グループ統合・回帰写像を一つずつ追加した結果を保持する。
Excel出力は reporting/scenario_excel.py が担当する。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import rank_to_unit_interval


SCENARIO_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "S00_Current_Direct_EW": {
        "title": "現行モデル：0-1順位の直接等ウェイト",
        "description": (
            "各ファクターを時点別に0-1順位へ変換し、欠損を0として全ファクターを固定分母で平均します。"
            "現在運用している単純合算モデルを再現する比較基準です。"
        ),
    },
    "S01_Missing_Adjusted_EW": {
        "title": "方向・欠損調整後の直接等ウェイト",
        "description": (
            "ファクター方向を統一し、銘柄ごとに利用可能なファクターだけでウェイトを再正規化します。"
            "欠損の多い銘柄が機械的に低スコアとなる影響を切り分けます。"
        ),
    },
    "S02_Winsorized_Direct_EW": {
        "title": "外れ値処理後の直接等ウェイト",
        "description": (
            "指標固有変換とWinsorize後の値を0-1順位化して直接平均します。"
            "外れ値処理だけを追加した増分効果を確認します。"
        ),
    },
    "S03_Neutralized_Direct_EW": {
        "title": "国・セクター・サイズ中立化後の直接等ウェイト",
        "description": (
            "外れ値処理後に国・セクター・時価総額を中立化し、残差を0-1順位化して直接平均します。"
            "構造要因を除いた個別銘柄選別能力を確認します。"
        ),
    },
    "S04_Hierarchical_Equal_Weight": {
        "title": "階層等ウェイト",
        "description": (
            "中立化後ファクターをグループ内で等ウェイトし、Value・Momentum等のグループを再度等ウェイトします。"
            "ファクター数の多いグループが過大な影響を持つ問題を修正します。"
        ),
    },
    "S05_Correlation_Adjusted_IC": {
        "title": "相関調整ICウェイト",
        "description": (
            "中立化後ファクタースコアについて、過去RankICとファクター相関からグループ内ウェイトを推定し、"
            "グループ間は等ウェイトします。時点tのウェイトにはt-1以前だけを使用します。"
        ),
    },
    "S06_Selected_Factor_Models": {
        "title": "単一ファクターモデル選択＋階層等ウェイト",
        "description": (
            "設定された原系列・差分・移動平均乖離等をそれぞれ1ファクターとして扱い、"
            "Linear・Piecewise・Quadratic・Combined RidgeからOOSで選択します。"
            "予測値を0-1スコアへ変換後、グループ内・グループ間を等ウェイトします。"
        ),
    },
    "S07_Full_OOF_Ridge": {
        "title": "最終候補：設定済みグループ統合＋OOF Ridge",
        "description": (
            "原系列と設定済み派生ファクターの選択済み単一ファクターモデル、"
            "Excelで指定したグループ統合方法、グループ間OOF Ridgeを順に適用したフルモデルです。"
        ),
    },
}


@dataclass
class ScenarioResult:
    scenario_id: str
    title: str
    description: str
    stock_scores: pd.DataFrame
    factor_values: pd.DataFrame
    group_scores: pd.DataFrame
    factor_weights: pd.DataFrame


def _rank_01_by_date(df: pd.DataFrame, source_col: str, date_col: str) -> pd.Series:
    return df.groupby(date_col, group_keys=False)[source_col].transform(rank_to_unit_interval)


def _add_score_fields(df: pd.DataFrame, score_col: str, date_col: str, score_scale: str = "zero_one") -> pd.DataFrame:
    out = df.copy()
    raw = pd.to_numeric(out[score_col], errors="coerce")
    if score_scale == "minus1_1":
        out["stock_score_minus1_1"] = raw.clip(-1.0, 1.0)
        out["stock_score_0_1"] = (out["stock_score_minus1_1"] + 1.0) / 2.0
    else:
        out["stock_score_0_1"] = raw.clip(0.0, 1.0)
        out["stock_score_minus1_1"] = 2.0 * out["stock_score_0_1"] - 1.0
    out["score_rank_pct"] = out.groupby(date_col, group_keys=False)["stock_score_0_1"].transform(rank_to_unit_interval)

    def quintile(s: pd.Series) -> pd.Series:
        valid = s.notna()
        result = pd.Series(pd.NA, index=s.index, dtype="Int64")
        if valid.sum() < 5 or s.loc[valid].nunique() < 2:
            return result
        ranks = s.loc[valid].rank(method="first", pct=True)
        result.loc[valid] = np.ceil(ranks * 5).clip(1, 5).astype(int)
        return result

    quintiles = []
    for _, idx in out.groupby(date_col).groups.items():
        quintiles.append(quintile(out.loc[idx, "stock_score_0_1"]))
    out["score_quintile"] = pd.concat(quintiles).sort_index() if quintiles else pd.Series(pd.NA, index=out.index, dtype="Int64")
    return out


def _base_identifiers(panel: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    cols = config["columns"]
    candidates = [
        cols["date"], cols["isin"], cols.get("country"), cols.get("sector"),
        cols.get("currency"), cols.get("market_cap"), cols.get("stock_return"), "forward_return",
    ]
    return [c for c in candidates if c and c in panel.columns]


def _wide_factor_values(
    panel: pd.DataFrame,
    source_cols: dict[str, str],
    date_col: str,
    isin_col: str,
    id_cols: list[str],
    directions: dict[str, int],
    rank_values: bool = True,
) -> pd.DataFrame:
    out = panel[id_cols].copy()
    for factor, source in source_cols.items():
        values = pd.to_numeric(panel[source], errors="coerce") * int(directions.get(factor, 1))
        if rank_values:
            temp = pd.DataFrame({date_col: panel[date_col], "value": values}, index=panel.index)
            out[factor] = _rank_01_by_date(temp, "value", date_col)
        else:
            out[factor] = values
    return out


def _direct_average(
    factor_values: pd.DataFrame,
    factor_cols: list[str],
    fixed_denominator: bool,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    values = factor_values[factor_cols].apply(pd.to_numeric, errors="coerce")
    available = values.notna().sum(axis=1)
    coverage = available / max(len(factor_cols), 1)
    if fixed_denominator:
        score = values.fillna(0.0).sum(axis=1) / max(len(factor_cols), 1)
    else:
        score = values.mean(axis=1, skipna=True)
    return score, available, coverage


def _hierarchical_equal(
    factor_values: pd.DataFrame,
    factor_master: pd.DataFrame,
    date_col: str,
    isin_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    enabled = factor_master[factor_master["Enabled"].astype(int).eq(1)].copy()
    groups: dict[str, list[str]] = {
        str(group): [f for f in g["Factor_Code"].astype(str) if f in factor_values.columns]
        for group, g in enabled.groupby("Factor_Group")
    }
    group_df = factor_values[[date_col, isin_col]].copy()
    for group, factors in groups.items():
        if factors:
            group_df[group] = factor_values[factors].mean(axis=1, skipna=True)
    group_cols = [c for c in group_df.columns if c not in [date_col, isin_col]]
    final_score = group_df[group_cols].mean(axis=1, skipna=True) if group_cols else pd.Series(np.nan, index=group_df.index)
    return group_df, final_score


def _factor_long_from_wide(
    factor_values: pd.DataFrame,
    factor_master: pd.DataFrame,
    panel: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    date_col, isin_col = config["columns"]["date"], config["columns"]["isin"]
    factors = [f for f in factor_master.loc[factor_master["Enabled"].astype(int).eq(1), "Factor_Code"].astype(str) if f in factor_values]
    long = factor_values[[date_col, isin_col] + factors].melt(
        id_vars=[date_col, isin_col], var_name="factor", value_name="score_0_1"
    )
    long["prediction_score"] = 2.0 * long["score_0_1"] - 1.0
    # build_group_oof uses prediction for historical RankIC and prediction_score for aggregation.
    long["prediction"] = long["prediction_score"]
    targets = panel[[date_col, isin_col, "forward_return"]].drop_duplicates([date_col, isin_col])
    long = long.merge(targets, on=[date_col, isin_col], how="left")
    meta = factor_master[["Factor_Code", "Factor_Group"]].rename(columns={"Factor_Code": "factor", "Factor_Group": "group"})
    return long.merge(meta, on="factor", how="left")


def _normalize_nonnegative_with_cap(weights: pd.Series, cap: float) -> pd.Series:
    w = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if w.sum() <= 0:
        w[:] = 1.0
    w = w / w.sum()
    if cap >= 1.0 or len(w) == 1:
        return w
    result = pd.Series(0.0, index=w.index)
    free = pd.Series(True, index=w.index)
    remaining = 1.0
    for _ in range(len(w) + 2):
        if not free.any():
            break
        alloc = w[free]
        if alloc.sum() <= 0:
            alloc[:] = 1.0
        alloc = alloc / alloc.sum() * remaining
        over = alloc > cap
        if not over.any():
            result.loc[free] = alloc
            remaining = 0.0
            break
        capped = alloc.index[over]
        result.loc[capped] = cap
        remaining -= cap * len(capped)
        free.loc[capped] = False
    if remaining > 1e-8 and free.any():
        result.loc[free] += remaining / free.sum()
    return result / result.sum()


def _group_equal_from_factor_wide(
    factor_wide: pd.DataFrame,
    factor_master: pd.DataFrame,
    date_col: str,
    isin_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_wide, _ = _hierarchical_equal(factor_wide, factor_master, date_col, isin_col)
    dates = sorted(factor_wide[date_col].dropna().unique())
    rows = []
    enabled = factor_master[factor_master["Enabled"].astype(int).eq(1)]
    for group, g in enabled.groupby("Factor_Group"):
        factors = [f for f in g["Factor_Code"].astype(str) if f in factor_wide.columns]
        if not factors:
            continue
        weight = 1.0 / len(factors)
        for date in dates:
            for factor in factors:
                rows.append({
                    "date": date, "group": group, "factor": factor,
                    "method": "equal_weight", "weight": weight,
                    "fallback_used": False,
                })
    return group_wide, pd.DataFrame(rows)


def _ic_adjusted_group_scenario(
    factor_values: pd.DataFrame,
    factor_master: pd.DataFrame,
    group_settings: pd.DataFrame,
    method_params: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_col, isin_col = config["columns"]["date"], config["columns"]["isin"]
    factors = [f for f in factor_master.loc[factor_master["Enabled"].astype(int).eq(1), "Factor_Code"].astype(str) if f in factor_values]
    base = factor_values[[date_col, isin_col, "forward_return"] + factors].copy()
    dates = sorted(base[date_col].dropna().unique())
    ic_rows = []
    for factor in factors:
        for date, g in base[[date_col, factor, "forward_return"]].groupby(date_col):
            valid = g[[factor, "forward_return"]].dropna()
            ic = valid[factor].corr(valid["forward_return"], method="spearman") if len(valid) >= 3 else np.nan
            ic_rows.append({"date": date, "factor": factor, "rank_ic": ic})
    ic_df = pd.DataFrame(ic_rows)
    defaults = config["model"].get("group_method_defaults", {})
    param_lookup: dict[str, dict[str, Any]] = {}
    if method_params is not None and not method_params.empty:
        for group, g in method_params.groupby("Factor_Group"):
            param_lookup[str(group)] = {str(r["Param_Name"]): r["Param_Value"] for _, r in g.iterrows()}
    setting_lookup = group_settings.set_index("Factor_Group").to_dict(orient="index")
    group_frame = base[[date_col, isin_col]].copy()
    weight_rows: list[dict[str, Any]] = []

    for group, gmeta in factor_master[factor_master["Enabled"].astype(int).eq(1)].groupby("Factor_Group"):
        gfactors = [f for f in gmeta["Factor_Code"].astype(str) if f in factors]
        if not gfactors:
            continue
        settings = setting_lookup.get(group, {})
        lookback = int(settings.get("Lookback_Periods", 36))
        min_periods = int(settings.get("Min_Periods", 18))
        max_weight = float(settings.get("Max_Weight", 0.50))
        smoothing = float(settings.get("Weight_Smoothing", 0.50))
        params = {**defaults, **param_lookup.get(str(group), {})}
        halflife = float(params.get("ewm_halflife", 12))
        positive_only = bool(params.get("positive_ic_only", True))
        shrink = float(params.get("correlation_shrinkage", 0.20))
        previous = pd.Series(1.0 / len(gfactors), index=gfactors, dtype=float)
        score_parts = []

        for date in dates:
            hist_dates = [d for d in dates if d < date][-lookback:]
            fallback = len(hist_dates) < min_periods
            if fallback:
                weights = pd.Series(1.0 / len(gfactors), index=gfactors, dtype=float)
                mean_ic = pd.Series(np.nan, index=gfactors, dtype=float)
            else:
                mean_ic_dict = {}
                for factor in gfactors:
                    series = ic_df[(ic_df["factor"].eq(factor)) & (ic_df["date"].isin(hist_dates))].sort_values("date")["rank_ic"].dropna()
                    value = float(series.ewm(halflife=halflife, adjust=False).mean().iloc[-1]) if len(series) else 0.0
                    mean_ic_dict[factor] = max(value, 0.0) if positive_only else value
                mean_ic = pd.Series(mean_ic_dict, dtype=float)
                hist = base[base[date_col].isin(hist_dates)].set_index([date_col, isin_col])[gfactors]
                corr = hist.corr(min_periods=max(10, 2 * len(gfactors))).reindex(index=gfactors, columns=gfactors).fillna(0.0)
                np.fill_diagonal(corr.values, 1.0)
                matrix = (1.0 - shrink) * corr.to_numpy(dtype=float) + shrink * np.eye(len(gfactors))
                try:
                    raw = np.linalg.solve(matrix + 1e-8 * np.eye(len(gfactors)), mean_ic.to_numpy(dtype=float))
                except np.linalg.LinAlgError:
                    raw = mean_ic.to_numpy(dtype=float)
                weights = _normalize_nonnegative_with_cap(pd.Series(raw, index=gfactors), max_weight)
            weights = _normalize_nonnegative_with_cap(smoothing * previous + (1.0 - smoothing) * weights, max_weight)
            previous = weights.copy()
            current = base[base[date_col].eq(date)][[date_col, isin_col] + gfactors].copy()
            values = current[gfactors].to_numpy(dtype=float)
            mask = np.isfinite(values)
            w = weights.to_numpy(dtype=float)
            numerator = np.nansum(values * w, axis=1)
            denominator = np.sum(mask * w, axis=1)
            current[group] = np.divide(numerator, denominator, out=np.full(len(current), np.nan), where=denominator > 0)
            score_parts.append(current[[date_col, isin_col, group]])
            for factor in gfactors:
                weight_rows.append({
                    "date": date, "group": group, "factor": factor,
                    "method": "ic_adjusted", "weight": float(weights[factor]),
                    "mean_rank_ic": float(mean_ic.get(factor, np.nan)),
                    "fallback_used": fallback,
                })
        if score_parts:
            group_score = pd.concat(score_parts, ignore_index=True)
            group_frame = group_frame.merge(group_score, on=[date_col, isin_col], how="left")
    return group_frame, pd.DataFrame(weight_rows)



def _pivot_factor_predictions(
    factor_predictions: pd.DataFrame,
    date_col: str,
    isin_col: str,
) -> pd.DataFrame:
    pivot = factor_predictions.pivot_table(
        index=[date_col, isin_col], columns="factor", values="prediction_score", aggfunc="last"
    ).reset_index()
    factor_cols = [c for c in pivot.columns if c not in [date_col, isin_col]]
    pivot[factor_cols] = (pivot[factor_cols] + 1.0) / 2.0
    return pivot

def _group_wide_and_final(
    group_long: pd.DataFrame,
    date_col: str,
    isin_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    group_wide = group_long.pivot_table(
        index=[date_col, isin_col], columns="group", values="group_prediction", aggfunc="last"
    ).reset_index()
    group_cols = [c for c in group_wide.columns if c not in [date_col, isin_col]]
    group_wide[group_cols] = (group_wide[group_cols] + 1.0) / 2.0
    final = group_wide[group_cols].mean(axis=1, skipna=True) if group_cols else pd.Series(np.nan, index=group_wide.index)
    return group_wide, final


def _merge_attributes(frame: pd.DataFrame, panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    date_col, isin_col = config["columns"]["date"], config["columns"]["isin"]
    id_cols = _base_identifiers(panel, config)
    attrs = panel[id_cols].drop_duplicates([date_col, isin_col])
    keep = [date_col, isin_col] + [c for c in id_cols if c not in frame.columns and c not in [date_col, isin_col]]
    return attrs[keep].merge(frame, on=[date_col, isin_col], how="right")


def _filter_scope(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    settings = config["report"].get("scenario_excel", {})
    date_col = config["columns"]["date"]
    scope = str(settings.get("date_scope", "all")).lower()
    if df.empty or date_col not in df:
        return df
    if scope == "latest":
        return df[df[date_col].eq(df[date_col].max())].copy()
    if scope == "selected":
        dates = pd.to_datetime(settings.get("selected_dates", []), errors="coerce")
        return df[df[date_col].isin(dates)].copy()
    return df.copy()


def build_stock_scoring_scenarios(
    panel: pd.DataFrame,
    factors: list[str],
    factor_master: pd.DataFrame,
    base_factor_master: pd.DataFrame,
    group_settings: pd.DataFrame,
    method_params: pd.DataFrame,
    factor_oof: pd.DataFrame,
    latest_factor: pd.DataFrame,
    group_oof: pd.DataFrame,
    latest_group: pd.DataFrame,
    composite_oof: pd.DataFrame,
    latest_stock: pd.DataFrame,
    group_weights_full: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, ScenarioResult]:
    """Build all enabled stock-score comparison scenarios."""
    date_col, isin_col = config["columns"]["date"], config["columns"]["isin"]
    id_cols = _base_identifiers(panel, config)
    fm = factor_master[factor_master["Enabled"].astype(int).eq(1)].copy()
    base_fm = base_factor_master[base_factor_master["Enabled"].astype(int).eq(1)].copy()
    factor_list = [f for f in base_fm["Factor_Code"].astype(str) if f in factors]
    directions = base_fm.set_index("Factor_Code")["Direction"].astype(int).to_dict()
    results: dict[str, ScenarioResult] = {}

    transformed_map = {f: f"{f}__transformed" for f in factor_list}
    winsor_map = {f: f"{f}__win" for f in factor_list}
    neutral_map = {f: f"{f}__neutral" for f in factor_list}

    raw_rank = _wide_factor_values(panel, transformed_map, date_col, isin_col, id_cols, directions)
    win_rank = _wide_factor_values(panel, winsor_map, date_col, isin_col, id_cols, directions)
    neutral_rank = _wide_factor_values(panel, neutral_map, date_col, isin_col, id_cols, directions)

    def add_simple(sid: str, factor_values: pd.DataFrame, fixed: bool, hierarchical: bool = False) -> None:
        if hierarchical:
            group_df, score = _hierarchical_equal(factor_values, base_fm, date_col, isin_col)
        else:
            group_df = pd.DataFrame(columns=[date_col, isin_col])
            score, _, _ = _direct_average(factor_values, factor_list, fixed)
        _, available, coverage = _direct_average(factor_values, factor_list, fixed)
        stock = factor_values[id_cols].copy()
        stock["factor_count_available"] = available
        stock["factor_coverage"] = coverage
        stock["score_before_ranking"] = score
        stock = _add_score_fields(stock, "score_before_ranking", date_col)
        if hierarchical and not group_df.empty:
            stock = stock.merge(group_df, on=[date_col, isin_col], how="left")
        info = SCENARIO_DESCRIPTIONS[sid]
        results[sid] = ScenarioResult(
            sid, info["title"], info["description"],
            stock, factor_values, group_df, pd.DataFrame(),
        )

    add_simple("S00_Current_Direct_EW", raw_rank, fixed=True)
    add_simple("S01_Missing_Adjusted_EW", raw_rank, fixed=False)
    add_simple("S02_Winsorized_Direct_EW", win_rank, fixed=False)
    add_simple("S03_Neutralized_Direct_EW", neutral_rank, fixed=False)
    add_simple("S04_Hierarchical_Equal_Weight", neutral_rank, fixed=False, hierarchical=True)

    # S05: neutralized rank scores + correlation-adjusted IC weights + equal group mean.
    ic_input = neutral_rank.copy()
    if "forward_return" not in ic_input.columns:
        ic_input = ic_input.merge(panel[[date_col, isin_col, "forward_return"]], on=[date_col, isin_col], how="left")
    ic_group_wide, ic_weights = _ic_adjusted_group_scenario(
        ic_input, base_fm, group_settings, method_params, config
    )
    ic_group_cols = [c for c in ic_group_wide.columns if c not in [date_col, isin_col]]
    ic_final = ic_group_wide[ic_group_cols].mean(axis=1, skipna=True)
    ic_factor_wide = neutral_rank.copy()
    ic_frame = ic_group_wide.copy()
    ic_frame["score_before_ranking"] = ic_final.to_numpy()
    ic_stock = _merge_attributes(ic_frame, panel, config)
    ic_stock = _add_score_fields(ic_stock, "score_before_ranking", date_col)
    info = SCENARIO_DESCRIPTIONS["S05_Correlation_Adjusted_IC"]
    results["S05_Correlation_Adjusted_IC"] = ScenarioResult(
        "S05_Correlation_Adjusted_IC", info["title"], info["description"],
        ic_stock, ic_factor_wide, ic_group_wide, ic_weights,
    )

    # Common factor model predictions for S06/S07: OOS history + latest fit.
    all_factor_predictions = pd.concat([factor_oof, latest_factor], ignore_index=True, sort=False)
    all_factor_predictions = all_factor_predictions.drop_duplicates([date_col, isin_col, "factor"], keep="last")
    model_factor_wide = _pivot_factor_predictions(all_factor_predictions, date_col, isin_col)

    # S06: selected single-factor models, equal weight inside and across groups.
    eq_group_wide, eq_weights = _group_equal_from_factor_wide(
        model_factor_wide, fm, date_col, isin_col
    )
    eq_group_cols = [c for c in eq_group_wide.columns if c not in [date_col, isin_col]]
    eq_final = eq_group_wide[eq_group_cols].mean(axis=1, skipna=True)
    eq_final_frame = eq_group_wide[[date_col, isin_col]].copy()
    eq_final_frame["score_before_ranking"] = eq_final.values
    eq_stock = _merge_attributes(eq_final_frame.merge(eq_group_wide, on=[date_col, isin_col]), panel, config)
    eq_stock = _add_score_fields(eq_stock, "score_before_ranking", date_col)
    info = SCENARIO_DESCRIPTIONS["S06_Selected_Factor_Models"]
    results["S06_Selected_Factor_Models"] = ScenarioResult(
        "S06_Selected_Factor_Models", info["title"], info["description"],
        eq_stock, _merge_attributes(model_factor_wide, panel, config), eq_group_wide, eq_weights,
    )

    # S07: existing full OOF model and latest score.
    full_stock = pd.concat([
        composite_oof,
        latest_stock,
    ], ignore_index=True, sort=False).drop_duplicates([date_col, isin_col], keep="last")
    full_stock["score_before_ranking"] = full_stock.get("stock_alpha", np.nan)
    if "stock_score" in full_stock:
        full_stock = _add_score_fields(full_stock, "stock_score", date_col, score_scale="minus1_1")
    else:
        full_stock = _add_score_fields(full_stock, "score_before_ranking", date_col)
    full_stock = _merge_attributes(full_stock, panel, config)
    full_group = pd.concat([group_oof, latest_group], ignore_index=True, sort=False).drop_duplicates(
        [date_col, isin_col, "group"], keep="last"
    )
    full_group_wide, _ = _group_wide_and_final(full_group, date_col, isin_col)
    info = SCENARIO_DESCRIPTIONS["S07_Full_OOF_Ridge"]
    results["S07_Full_OOF_Ridge"] = ScenarioResult(
        "S07_Full_OOF_Ridge", info["title"], info["description"],
        full_stock, _merge_attributes(model_factor_wide, panel, config), full_group_wide, group_weights_full,
    )

    return results
