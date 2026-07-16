from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV

from .master import FactorMeta
from .preprocessing import build_factor_scores, percentile_rank


@dataclass
class ScenarioResult:
    stock_scores: pd.DataFrame
    factor_scores: pd.DataFrame
    sub_scores: pd.DataFrame
    weight_history: pd.DataFrame
    model_selection: pd.DataFrame


def _rank_total(data: pd.DataFrame, date_col: str, pred_col: str = "Prediction") -> pd.Series:
    return data.groupby(date_col)[pred_col].transform(percentile_rank)


def _assign_quintile(s: pd.Series, q: int) -> pd.Series:
    valid = s.notna()
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    n = int(valid.sum())
    if n >= q:
        ranks = s.loc[valid].rank(method="first")
        out.loc[valid] = pd.qcut(ranks, q=q, labels=range(1, q + 1)).astype(int)
    return out


def _stock_frame(data: pd.DataFrame, prediction: pd.Series, config: dict[str, Any]) -> pd.DataFrame:
    c = config["columns"]
    out = pd.DataFrame({
        "Date": data[c["date"]],
        "ISIN": data[c["isin"]],
        "Currency": data[c["currency"]],
        "MarketCap": data[c["market_cap"]],
        "Prediction": prediction,
        "NextMonthReturn": data["NextMonthReturn"],
    })
    out["TotalScore"] = _rank_total(out, "Date")
    out["Quintile"] = out.groupby("Date", group_keys=False)["TotalScore"].apply(
        lambda s: _assign_quintile(s, int(config["evaluation"].get("quintiles", 5)))
    ).reset_index(level=0, drop=True)
    return out


def _factor_long(data: pd.DataFrame, scores: pd.DataFrame, metas: dict[str, FactorMeta], config: dict[str, Any]) -> pd.DataFrame:
    c = config["columns"]
    cols = [k for k in metas if k in scores.columns]
    if not cols:
        return pd.DataFrame(columns=["Date", "ISIN", "FactorCode", "FactorScore"])
    tmp = pd.concat([data[[c["date"], c["isin"]]].reset_index(drop=True), scores[cols].reset_index(drop=True)], axis=1)
    return tmp.melt(id_vars=[c["date"], c["isin"]], value_vars=cols, var_name="FactorCode", value_name="FactorScore").rename(
        columns={c["date"]: "Date", c["isin"]: "ISIN"}
    )


def _group_scores(scores: pd.DataFrame, metas: dict[str, FactorMeta], method: str = "equal") -> pd.DataFrame:
    groups: dict[str, list[str]] = {}
    for code, meta in metas.items():
        if code in scores:
            groups.setdefault(meta.group, []).append(code)
    out = pd.DataFrame(index=scores.index)
    for group, cols in groups.items():
        if method == "manual":
            weights = np.array([max(metas[c].base_weight, 0.0) for c in cols], dtype=float)
            if weights.sum() == 0:
                weights = np.ones(len(cols))
            arr = scores[cols].to_numpy(float)
            valid = np.isfinite(arr)
            num = np.nansum(arr * weights, axis=1)
            den = np.sum(valid * weights, axis=1)
            out[group] = np.divide(num, den, out=np.full(len(arr), np.nan), where=den > 0)
        else:
            out[group] = scores[cols].mean(axis=1, skipna=True)
    return out


def _subscore_long(data: pd.DataFrame, group_scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    c = config["columns"]
    if group_scores.empty:
        return pd.DataFrame(columns=["Date", "ISIN", "SubScore", "SubScoreValue"])
    tmp = pd.concat([data[[c["date"], c["isin"]]].reset_index(drop=True), group_scores.reset_index(drop=True)], axis=1)
    return tmp.melt(id_vars=[c["date"], c["isin"]], var_name="SubScore", value_name="SubScoreValue").rename(
        columns={c["date"]: "Date", c["isin"]: "ISIN"}
    )


def _monthly_ic(data: pd.DataFrame, scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    c = config["columns"]
    rows = []
    for date, idx in data.groupby(c["date"]).groups.items():
        y = data.loc[idx, "NextMonthReturn"]
        for code in scores.columns:
            x = scores.loc[idx, code]
            mask = x.notna() & y.notna()
            ic = spearmanr(x[mask], y[mask]).statistic if mask.sum() >= 8 else np.nan
            rows.append({"Date": date, "FactorCode": code, "RankIC": ic})
    return pd.DataFrame(rows)


def _correlation_adjusted_group_scores(
    data: pd.DataFrame,
    scores: pd.DataFrame,
    metas: dict[str, FactorMeta],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = config["columns"]
    lookback = int(config["model"].get("ic_lookback_periods", 36))
    minp = int(config["model"].get("ic_minimum_periods", 12))
    ic = _monthly_ic(data, scores, config)
    dates = sorted(pd.to_datetime(data[c["date"]].dropna().unique()))
    groups: dict[str, list[str]] = {}
    for code, meta in metas.items():
        if code in scores:
            groups.setdefault(meta.group, []).append(code)
    result = pd.DataFrame(index=data.index)
    weight_rows: list[dict[str, Any]] = []

    for group, cols in groups.items():
        values = pd.Series(np.nan, index=data.index, dtype=float)
        for date in dates:
            idx = data.index[data[c["date"]] == date]
            past_dates = [d for d in dates if d < date][-lookback:]
            hist = ic[(ic["Date"].isin(past_dates)) & (ic["FactorCode"].isin(cols))]
            means = hist.groupby("FactorCode")["RankIC"].mean().reindex(cols)
            counts = hist.groupby("FactorCode")["RankIC"].count().reindex(cols).fillna(0)
            mu = means.clip(lower=0).fillna(0).to_numpy()
            if (counts < minp).any() or mu.sum() <= 0:
                w = np.ones(len(cols)) / len(cols)
                reason = "fallback_equal_weight"
            else:
                corr = scores.loc[idx, cols].corr().fillna(0).to_numpy()
                corr = 0.8 * corr + 0.2 * np.eye(len(cols))
                try:
                    raw = np.linalg.pinv(corr) @ mu
                    raw = np.clip(raw, 0, None)
                    w = raw / raw.sum() if raw.sum() > 0 else np.ones(len(cols)) / len(cols)
                    reason = "correlation_adjusted_ic"
                except np.linalg.LinAlgError:
                    w = np.ones(len(cols)) / len(cols)
                    reason = "fallback_equal_weight"
            arr = scores.loc[idx, cols].to_numpy(float)
            valid = np.isfinite(arr)
            num = np.nansum(arr * w, axis=1)
            den = np.sum(valid * w, axis=1)
            values.loc[idx] = np.divide(num, den, out=np.full(len(idx), np.nan), where=den > 0)
            for code, ww in zip(cols, w):
                weight_rows.append({"Date": date, "Factor_Group": group, "FactorCode": code, "Weight": ww, "Reason": reason})
        result[group] = values
    return result, pd.DataFrame(weight_rows)


def _design(x: np.ndarray, model_name: str, knot: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if model_name == "linear":
        return x[:, None]
    if model_name == "piecewise":
        return np.column_stack([x, np.maximum(x - knot, 0.0)])
    if model_name == "quadratic":
        return np.column_stack([x, x**2])
    if model_name == "combined_ridge":
        return np.column_stack([x, np.maximum(x - knot, 0.0), x**2])
    raise ValueError(model_name)


def _fit_candidate(x: np.ndarray, y: np.ndarray, model_name: str, alphas: list[float]):
    X = _design(x, model_name)
    model = RidgeCV(alphas=alphas) if model_name == "combined_ridge" else LinearRegression()
    model.fit(X, y)
    return model


def _mean_monthly_rank_ic(dates: pd.Series, pred: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    tmp = pd.DataFrame({"Date": dates.to_numpy(), "Prediction": pred, "Return": y})
    vals = []
    for _, g in tmp.groupby("Date"):
        m = g[["Prediction", "Return"]].dropna()
        if len(m) >= 8:
            vals.append(spearmanr(m["Prediction"], m["Return"]).statistic)
    if not vals:
        return np.nan, np.nan, 0
    arr = np.asarray(vals, float)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else np.nan, len(arr)


def rolling_selected_factor_models(
    data: pd.DataFrame,
    scores: pd.DataFrame,
    metas: dict[str, FactorMeta],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """時間順CVで4候補を比較し、1-SEルールで単純モデルを選択する参照実装。"""
    c = config["columns"]
    dates = sorted(pd.to_datetime(data[c["date"]].dropna().unique()))
    window = int(config["model"].get("training_window_periods", 36))
    min_train = int(config["model"].get("minimum_train_periods", 18))
    models = list(config["model"].get("candidate_models", ["linear", "piecewise", "quadratic", "combined_ridge"]))
    alphas = list(config["model"].get("ridge_alphas", [0.1, 1, 10]))
    complexity = {"linear": 1, "piecewise": 2, "quadratic": 2, "combined_ridge": 3}
    predictions = pd.DataFrame(index=data.index)
    rows: list[dict[str, Any]] = []

    for code in [k for k in metas if k in scores]:
        pred_series = pd.Series(np.nan, index=data.index, dtype=float)
        for pos, date in enumerate(dates):
            train_dates = dates[max(0, pos - window):pos]
            if len(train_dates) < min_train:
                continue
            val_n = max(6, min(12, len(train_dates) // 3))
            fit_dates, val_dates = train_dates[:-val_n], train_dates[-val_n:]
            fit_idx = data.index[data[c["date"]].isin(fit_dates)]
            val_idx = data.index[data[c["date"]].isin(val_dates)]
            test_idx = data.index[data[c["date"]] == date]
            fit_mask = scores.loc[fit_idx, code].notna() & data.loc[fit_idx, "NextMonthReturn"].notna()
            val_mask = scores.loc[val_idx, code].notna() & data.loc[val_idx, "NextMonthReturn"].notna()
            if fit_mask.sum() < 100 or val_mask.sum() < 50:
                continue
            candidates = []
            for model_name in models:
                model = _fit_candidate(
                    scores.loc[fit_idx[fit_mask], code].to_numpy(),
                    data.loc[fit_idx[fit_mask], "NextMonthReturn"].to_numpy(),
                    model_name,
                    alphas,
                )
                vp = model.predict(_design(scores.loc[val_idx[val_mask], code].to_numpy(), model_name))
                mean_ic, se, n_eval = _mean_monthly_rank_ic(
                    data.loc[val_idx[val_mask], c["date"]], vp, data.loc[val_idx[val_mask], "NextMonthReturn"].to_numpy()
                )
                candidates.append((model_name, mean_ic, se, n_eval))
            valid = [x for x in candidates if np.isfinite(x[1])]
            if not valid:
                continue
            best = max(valid, key=lambda z: z[1])
            threshold = best[1] - (best[2] if config["model"].get("one_se_rule", True) and np.isfinite(best[2]) else 0.0)
            eligible = [x for x in valid if x[1] >= threshold]
            selected = min(eligible, key=lambda z: (complexity[z[0]], -z[1]))
            full_idx = data.index[data[c["date"]].isin(train_dates)]
            full_mask = scores.loc[full_idx, code].notna() & data.loc[full_idx, "NextMonthReturn"].notna()
            test_mask = scores.loc[test_idx, code].notna()
            final_model = _fit_candidate(
                scores.loc[full_idx[full_mask], code].to_numpy(),
                data.loc[full_idx[full_mask], "NextMonthReturn"].to_numpy(),
                selected[0], alphas,
            )
            pred_series.loc[test_idx[test_mask]] = final_model.predict(
                _design(scores.loc[test_idx[test_mask], code].to_numpy(), selected[0])
            )
            rows.append({
                "Date": date,
                "FactorCode": code,
                "SelectedModel": selected[0],
                "BestRawModel": best[0],
                "SelectedMeanRankIC": selected[1],
                "BestMeanRankIC": best[1],
                "BestSE": best[2],
                "OneSEThreshold": threshold,
                "SelectionReason": "best_metric" if selected[0] == best[0] else "one_se_simpler_model",
            })
        predictions[code] = pred_series
    return predictions, pd.DataFrame(rows)


def _finalize(
    name: str,
    data: pd.DataFrame,
    scores: pd.DataFrame,
    groups: pd.DataFrame,
    prediction: pd.Series,
    metas: dict[str, FactorMeta],
    config: dict[str, Any],
    weight_history: pd.DataFrame | None = None,
    model_selection: pd.DataFrame | None = None,
) -> ScenarioResult:
    stock = _stock_frame(data, prediction, config)
    stock.insert(0, "Scenario", name)

    # 個別FA/SubScoreのExcel出力がlatestの場合、保持データも最新時点に限定する。
    # 評価に必要なstock_scoresは全期間を維持する。
    scenario_output = config.get("outputs", {}).get("scenario_excel", {})
    if scenario_output.get("date_scope", "latest") == "latest" and not data.empty:
        latest = pd.to_datetime(data[config["columns"]["date"]]).max()
        output_mask = pd.to_datetime(data[config["columns"]["date"]]).eq(latest)
        output_data = data.loc[output_mask]
        output_scores = scores.loc[output_mask]
        output_groups = groups.loc[output_mask]
    else:
        output_data, output_scores, output_groups = data, scores, groups

    return ScenarioResult(
        stock_scores=stock,
        factor_scores=_factor_long(output_data, output_scores, metas, config),
        sub_scores=_subscore_long(output_data, output_groups, config),
        weight_history=weight_history if weight_history is not None else pd.DataFrame(),
        model_selection=model_selection if model_selection is not None else pd.DataFrame(),
    )


def build_scenarios(
    data: pd.DataFrame,
    config: dict[str, Any],
    raw_metas: dict[str, FactorMeta],
    all_metas: dict[str, FactorMeta],
) -> dict[str, ScenarioResult]:
    c = config["columns"]
    enabled = config.get("scenarios", {})
    results: dict[str, ScenarioResult] = {}

    raw_scores = build_factor_scores(data, config, raw_metas, winsorize=False, neutralize=False, rank_transform="uniform_0_1")
    win_scores = build_factor_scores(data, config, raw_metas, winsorize=True, neutralize=False, rank_transform="uniform_0_1")
    neu_scores = build_factor_scores(data, config, raw_metas, winsorize=True, neutralize=True, rank_transform="uniform_0_1")

    raw_cols = [k for k in raw_metas if k in raw_scores]
    if enabled.get("S00_Current_Direct_EW", True):
        # 欠損は0.5で補完し、ファクター数固定で平均。
        pred = raw_scores[raw_cols].fillna(0.5).mean(axis=1)
        groups = _group_scores(raw_scores, raw_metas)
        results["S00_Current_Direct_EW"] = _finalize("S00_Current_Direct_EW", data, raw_scores, groups, pred, raw_metas, config)
    if enabled.get("S01_Missing_Adjusted_EW", True):
        pred = raw_scores[raw_cols].mean(axis=1, skipna=True)
        groups = _group_scores(raw_scores, raw_metas)
        results["S01_Missing_Adjusted_EW"] = _finalize("S01_Missing_Adjusted_EW", data, raw_scores, groups, pred, raw_metas, config)
    if enabled.get("S02_Winsorized_Direct_EW", True):
        pred = win_scores[raw_cols].mean(axis=1, skipna=True)
        groups = _group_scores(win_scores, raw_metas)
        results["S02_Winsorized_Direct_EW"] = _finalize("S02_Winsorized_Direct_EW", data, win_scores, groups, pred, raw_metas, config)
    if enabled.get("S03_Neutralized_Direct_EW", True):
        pred = neu_scores[raw_cols].mean(axis=1, skipna=True)
        groups = _group_scores(neu_scores, raw_metas)
        results["S03_Neutralized_Direct_EW"] = _finalize("S03_Neutralized_Direct_EW", data, neu_scores, groups, pred, raw_metas, config)
    if enabled.get("S04_Hierarchical_Equal_Weight", True):
        groups = _group_scores(neu_scores, raw_metas)
        pred = groups.mean(axis=1, skipna=True)
        results["S04_Hierarchical_Equal_Weight"] = _finalize("S04_Hierarchical_Equal_Weight", data, neu_scores, groups, pred, raw_metas, config)
    if enabled.get("S05_Correlation_Adjusted_IC", True):
        groups, wh = _correlation_adjusted_group_scores(data, neu_scores, raw_metas, config)
        pred = groups.mean(axis=1, skipna=True)
        results["S05_Correlation_Adjusted_IC"] = _finalize("S05_Correlation_Adjusted_IC", data, neu_scores, groups, pred, raw_metas, config, wh)

    if enabled.get("S06_Selected_Factor_Models", False) or enabled.get("S07_Full_OOF_Ridge", False):
        model_scores = build_factor_scores(data, config, all_metas, winsorize=True, neutralize=True, rank_transform="gaussian")
        factor_pred, selection = rolling_selected_factor_models(data, model_scores, all_metas, config)
        model_groups = _group_scores(factor_pred, all_metas)
        if enabled.get("S06_Selected_Factor_Models", False):
            pred = factor_pred.mean(axis=1, skipna=True)
            results["S06_Selected_Factor_Models"] = _finalize(
                "S06_Selected_Factor_Models", data, factor_pred, model_groups, pred, all_metas, config, model_selection=selection
            )
        if enabled.get("S07_Full_OOF_Ridge", False):
            dates = sorted(pd.to_datetime(data[c["date"]].unique()))
            pred = pd.Series(np.nan, index=data.index, dtype=float)
            alphas = config["model"].get("ridge_alphas", [0.1, 1, 10])
            min_train = int(config["model"].get("minimum_train_periods", 18))
            window = int(config["model"].get("training_window_periods", 36))
            for pos, date in enumerate(dates):
                train_dates = dates[max(0, pos - window):pos]
                if len(train_dates) < min_train:
                    continue
                tr = data.index[data[c["date"]].isin(train_dates)]
                te = data.index[data[c["date"]] == date]
                Xtr = model_groups.loc[tr]
                ytr = data.loc[tr, "NextMonthReturn"]
                mask = Xtr.notna().all(axis=1) & ytr.notna()
                if mask.sum() < 100:
                    continue
                model = RidgeCV(alphas=alphas).fit(Xtr.loc[mask], ytr.loc[mask])
                temask = model_groups.loc[te].notna().all(axis=1)
                pred.loc[te[temask]] = model.predict(model_groups.loc[te[temask]])
            results["S07_Full_OOF_Ridge"] = _finalize(
                "S07_Full_OOF_Ridge", data, factor_pred, model_groups, pred, all_metas, config, model_selection=selection
            )
    return results
