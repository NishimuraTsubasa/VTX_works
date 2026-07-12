from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.decomposition import PCA

from .factor_master import factor_lookup, group_lookup, method_param_lookup
from .utils import rank_to_unit_interval, safe_pearson, safe_spearman

LOGGER = logging.getLogger(__name__)


MODEL_COMPLEXITY = {
    "linear": 1,
    "piecewise": 2,
    "quadratic": 2,
    "combined_ridge": 3,
}


@dataclass
class ModelRunResult:
    predictions: pd.DataFrame
    metrics_by_date: pd.DataFrame
    metrics_summary: pd.DataFrame
    selection: pd.DataFrame
    selection_detail: pd.DataFrame
    coefficients: pd.DataFrame


def design_matrix(z: np.ndarray, model_name: str, knot: float = 0.0) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(-1)
    if model_name == "linear":
        return z[:, None]
    if model_name == "piecewise":
        return np.column_stack([z, np.maximum(z - knot, 0.0)])
    if model_name == "quadratic":
        return np.column_stack([z, z**2 - 1.0])
    if model_name == "combined_ridge":
        return np.column_stack([z, np.maximum(z - knot, 0.0), z**2 - 1.0])
    raise ValueError(f"Unknown model: {model_name}")


def _fit(model_name: str, z: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None, config: dict):
    X = design_matrix(z, model_name, float(config["diagnostics"]["piecewise_knot"]))
    if model_name == "combined_ridge":
        model = RidgeCV(alphas=np.asarray(config["model"]["ridge_alphas"], dtype=float), fit_intercept=True)
    else:
        model = LinearRegression(fit_intercept=True)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _predict(model, model_name: str, z: np.ndarray, config: dict) -> np.ndarray:
    X = design_matrix(z, model_name, float(config["diagnostics"]["piecewise_knot"]))
    return np.asarray(model.predict(X), dtype=float)


def _date_equal_weights(train: pd.DataFrame, date_col: str) -> np.ndarray:
    counts = train.groupby(date_col)[date_col].transform("count").to_numpy(dtype=float)
    return 1.0 / np.maximum(counts, 1.0)


def _top_bottom_spread(pred: pd.Series, actual: pd.Series, bins: int) -> float:
    valid = pd.DataFrame({"p": pred, "y": actual}).dropna()
    if len(valid) < max(10, bins * 2) or valid["p"].nunique() < bins:
        return np.nan
    try:
        q = pd.qcut(valid["p"], bins, labels=False, duplicates="drop")
        if q.nunique() < 2:
            return np.nan
        return float(valid.loc[q == q.max(), "y"].mean() - valid.loc[q == q.min(), "y"].mean())
    except ValueError:
        return np.nan


def walk_forward_single_factors(panel: pd.DataFrame, factor_map: dict[str, str], config: dict) -> ModelRunResult:
    cols = config["columns"]
    date_col, isin_col = cols["date"], cols["isin"]
    target_col = "forward_return"
    dates = sorted(panel.loc[panel[target_col].notna(), date_col].drop_duplicates())
    min_train = int(config["model"]["minimum_train_periods"])
    window = int(config["model"]["training_window_periods"])
    candidates = list(config["model"]["candidate_models"])
    pred_rows: list[pd.DataFrame] = []
    coef_rows: list[dict] = []

    for test_pos in range(min_train, len(dates)):
        test_date = dates[test_pos]
        train_dates = dates[max(0, test_pos - window) : test_pos]
        train_base = panel[panel[date_col].isin(train_dates)]
        test_base = panel[panel[date_col] == test_date]

        for factor, z_col in factor_map.items():
            train = train_base[[date_col, isin_col, z_col, target_col]].dropna()
            test = test_base[[date_col, isin_col, z_col, target_col]].copy()
            if len(train) < 50 or test[z_col].notna().sum() < 3:
                continue
            sw = _date_equal_weights(train, date_col) if config["model"].get("date_equal_weighting", True) else None
            for model_name in candidates:
                model = _fit(model_name, train[z_col].to_numpy(), train[target_col].to_numpy(), sw, config)
                valid_test = test[z_col].notna()
                pred = np.full(len(test), np.nan)
                pred[valid_test.to_numpy()] = _predict(model, model_name, test.loc[valid_test, z_col].to_numpy(), config)
                block = test[[date_col, isin_col, target_col]].copy()
                block["factor"] = factor
                block["model"] = model_name
                block["prediction"] = pred
                pred_rows.append(block)
                coefs = np.ravel(getattr(model, "coef_", []))
                coef_rows.append({
                    "train_end": train_dates[-1],
                    "test_date": test_date,
                    "factor": factor,
                    "model": model_name,
                    "intercept": float(getattr(model, "intercept_", np.nan)),
                    "coef_1": float(coefs[0]) if len(coefs) > 0 else np.nan,
                    "coef_2": float(coefs[1]) if len(coefs) > 1 else np.nan,
                    "coef_3": float(coefs[2]) if len(coefs) > 2 else np.nan,
                    "ridge_alpha": float(getattr(model, "alpha_", np.nan)),
                    "train_observations": len(train),
                })

    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    if predictions.empty:
        raise ValueError("No OOS predictions were generated. Check data length and minimum_train_periods.")

    metric_rows: list[dict] = []
    for (factor, model_name, date), g in predictions.groupby(["factor", "model", date_col]):
        metric_rows.append({
            "factor": factor,
            "model": model_name,
            "date": date,
            "rank_ic": safe_spearman(g["prediction"], g[target_col]),
            "pearson_ic": safe_pearson(g["prediction"], g[target_col]),
            "top_bottom_spread": _top_bottom_spread(
                g["prediction"], g[target_col], int(config["diagnostics"]["quantile_bins"])
            ),
            "observations": int(g[["prediction", target_col]].dropna().shape[0]),
        })
    metrics_by_date = pd.DataFrame(metric_rows)

    summary = (
        metrics_by_date.groupby(["factor", "model"])
        .agg(
            mean_rank_ic=("rank_ic", "mean"),
            std_rank_ic=("rank_ic", "std"),
            count_periods=("rank_ic", "count"),
            positive_rate=("rank_ic", lambda x: float((x > 0).mean())),
            mean_pearson_ic=("pearson_ic", "mean"),
            mean_top_bottom_spread=("top_bottom_spread", "mean"),
        )
        .reset_index()
    )
    summary["rank_ic_ir"] = summary["mean_rank_ic"] / summary["std_rank_ic"].replace(0, np.nan)
    summary["rank_ic_se"] = summary["std_rank_ic"] / np.sqrt(summary["count_periods"].clip(lower=1))
    selection, selection_detail = select_models(summary, config)
    return ModelRunResult(
        predictions, metrics_by_date, summary, selection, selection_detail, pd.DataFrame(coef_rows)
    )


def select_models(summary: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one model per factor and retain an auditable candidate-level decision table.

    Decision process
    ----------------
    1. Rank candidates by the configured primary OOS metric (default: mean RankIC).
    2. Define the one-standard-error threshold as
       best_metric - multiplier * best_model_standard_error.
    3. Treat every model above the threshold as statistically near-equivalent.
    4. Choose the least-complex model among eligible candidates; use the higher
       primary metric as the tie breaker.
    5. Apply factor-adoption gates after the model has been selected.
    """
    settings = config["model"].get("model_selection", {})
    metric_col = str(settings.get("primary_metric", "mean_rank_ic"))
    se_col = str(settings.get("standard_error_column", "rank_ic_se"))
    one_se = bool(settings.get("one_se_rule", config["model"].get("one_se_rule", True)))
    multiplier = float(settings.get("one_se_multiplier", 1.0))
    min_periods = int(settings.get("minimum_evaluation_periods", 1))
    complexity_map = {**MODEL_COMPLEXITY, **settings.get("complexity_order", {})}

    selection_rows: list[dict] = []
    detail_rows: list[dict] = []
    for factor, raw in summary.groupby("factor"):
        g = raw.copy()
        if metric_col not in g.columns:
            raise ValueError(f"Unknown model-selection metric: {metric_col}")
        g["complexity"] = g["model"].map(complexity_map).fillna(999).astype(int)
        g["evaluation_periods_pass"] = g["count_periods"].fillna(0).astype(int) >= min_periods
        valid = g[g["evaluation_periods_pass"] & g[metric_col].notna()].copy()
        if valid.empty:
            valid = g[g[metric_col].notna()].copy()
        if valid.empty:
            continue

        valid = valid.sort_values([metric_col, "complexity"], ascending=[False, True])
        best = valid.iloc[0]
        best_metric = float(best[metric_col])
        best_se = float(best[se_col]) if se_col in best.index and pd.notna(best[se_col]) else 0.0
        threshold = best_metric - multiplier * best_se if one_se else best_metric

        g["best_raw_model"] = str(best["model"])
        g["best_primary_metric"] = best_metric
        g["best_standard_error"] = best_se
        g["one_se_threshold"] = threshold
        g["delta_from_best"] = best_metric - g[metric_col]
        g["within_one_se"] = g["evaluation_periods_pass"] & g[metric_col].notna() & (g[metric_col] >= threshold)
        g["primary_metric_rank"] = g[metric_col].rank(method="min", ascending=False)

        eligible = g[g["within_one_se"]].copy() if one_se else g[g["model"].eq(best["model"])].copy()
        if eligible.empty:
            eligible = g[g["model"].eq(best["model"])].copy()
        selected = eligible.sort_values(["complexity", metric_col], ascending=[True, False]).iloc[0]
        selected_model = str(selected["model"])

        if selected_model == str(best["model"]):
            reason_code = "BEST_OOS_METRIC"
            reason_jp = "OOS平均RankICが最も高いため採用"
        elif one_se:
            reason_code = "ONE_SE_SIMPLER_MODEL"
            reason_jp = (
                "最良モデルとの差が1標準誤差以内で統計的に明確な差がないため、"
                "より単純なモデルを採用"
            )
        else:
            reason_code = "TIE_BREAKER"
            reason_jp = "同順位候補のうち設定したタイブレーク基準で採用"

        adoption = config["model"].get("factor_adoption", {})
        mean_ic_pass = float(selected.get("mean_rank_ic", np.nan)) >= float(adoption.get("minimum_mean_rank_ic", -1.0))
        positive_rate_pass = float(selected.get("positive_rate", np.nan)) >= float(adoption.get("minimum_positive_rate", 0.0))
        periods_pass = bool(selected.get("evaluation_periods_pass", True))
        adopted = bool(mean_ic_pass and positive_rate_pass and periods_pass)

        g["selected"] = g["model"].eq(selected_model)
        g["selected_model"] = selected_model
        g["selection_reason_code"] = reason_code
        g["selection_reason_jp"] = reason_jp
        g["adoption_mean_rank_ic_pass"] = mean_ic_pass
        g["adoption_positive_rate_pass"] = positive_rate_pass
        g["adoption_periods_pass"] = periods_pass
        g["adopted"] = adopted
        detail_rows.extend(g.to_dict("records"))

        row = selected.to_dict()
        row.update({
            "selected_model": selected_model,
            "best_raw_model": str(best["model"]),
            "primary_metric_name": metric_col,
            "selected_primary_metric": float(selected[metric_col]),
            "best_primary_metric": best_metric,
            "best_standard_error": best_se,
            "one_se_threshold": threshold,
            "selected_delta_from_best": best_metric - float(selected[metric_col]),
            "selected_complexity": int(selected["complexity"]),
            "one_se_rule_applied": one_se,
            "selection_reason_code": reason_code,
            "selection_reason_jp": reason_jp,
            "adoption_mean_rank_ic_pass": mean_ic_pass,
            "adoption_positive_rate_pass": positive_rate_pass,
            "adoption_periods_pass": periods_pass,
            "adopted": adopted,
        })
        selection_rows.append(row)

    return pd.DataFrame(selection_rows), pd.DataFrame(detail_rows)


def selected_oof_predictions(run: ModelRunResult, config: dict) -> pd.DataFrame:
    selected = run.selection[["factor", "selected_model", "adopted"]]
    pred = run.predictions.merge(selected, on="factor", how="inner")
    pred = pred[(pred["model"] == pred["selected_model"]) & pred["adopted"]].copy()
    date_col = config["columns"]["date"]
    pred["prediction_score"] = pred.groupby(["factor", date_col])["prediction"].transform(
        lambda s: 2.0 * rank_to_unit_interval(s) - 1.0
    )
    return pred


def _normalize_nonnegative_with_cap(weights: pd.Series, cap: float) -> pd.Series:
    w = weights.clip(lower=0).astype(float)
    if w.sum() <= 0:
        w[:] = 1.0
    w = w / w.sum()
    if cap >= 1.0 or len(w) <= 1:
        return w
    # Iterative water filling. If cap is infeasible, use equal weights.
    if cap * len(w) < 1.0 - 1e-10:
        return pd.Series(1.0 / len(w), index=w.index)
    free = pd.Series(True, index=w.index)
    result = pd.Series(0.0, index=w.index)
    remaining = 1.0
    source = w.copy()
    for _ in range(len(w) + 2):
        if not free.any():
            break
        alloc = source[free]
        if alloc.sum() <= 0:
            alloc[:] = 1.0
        alloc = alloc / alloc.sum() * remaining
        over = alloc > cap
        if not over.any():
            result.loc[free] = alloc
            remaining = 0.0
            break
        capped_idx = alloc.index[over]
        result.loc[capped_idx] = cap
        remaining -= cap * len(capped_idx)
        free.loc[capped_idx] = False
    if remaining > 1e-8 and free.any():
        result.loc[free] += remaining / free.sum()
    return result / result.sum()


def _normalize_signed(weights: pd.Series, cap: float) -> pd.Series:
    w = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if w.abs().sum() <= 0:
        w[:] = 1.0
    w = w / w.abs().sum()
    if cap < 1.0:
        w = w.clip(lower=-cap, upper=cap)
        if w.abs().sum() <= 0:
            w[:] = 1.0
        w = w / w.abs().sum()
    return w


def _fallback_weights(factors: list[str], base_weights: pd.Series, fallback: str) -> pd.Series:
    if fallback == "manual":
        return _normalize_nonnegative_with_cap(base_weights.reindex(factors).fillna(1.0), 1.0)
    return pd.Series(1.0 / len(factors), index=factors, dtype=float)


def _factor_ic_means(hist_metric: pd.DataFrame, factors: list[str], params: dict) -> pd.Series:
    method = str(params.get("ic_mean_method", "ewm")).lower()
    halflife = float(params.get("ewm_halflife", 12))
    values = {}
    for factor in factors:
        s = hist_metric.loc[hist_metric["factor"].eq(factor)].sort_values("date")["rank_ic"].dropna()
        if s.empty:
            values[factor] = np.nan
        elif method == "ewm":
            values[factor] = float(s.ewm(halflife=halflife, adjust=False).mean().iloc[-1])
        else:
            values[factor] = float(s.mean())
    out = pd.Series(values, dtype=float)
    if bool(params.get("positive_ic_only", True)):
        out = out.clip(lower=0)
    return out


def _ic_adjusted_weights(
    history: pd.DataFrame,
    hist_metric: pd.DataFrame,
    factors: list[str],
    params: dict,
    max_weight: float,
) -> tuple[pd.Series, pd.Series]:
    mu = _factor_ic_means(hist_metric, factors, params).fillna(0.0)
    pivot = history.pivot_table(index=["date", "ISIN"], columns="factor", values="prediction_score")
    pivot = pivot.reindex(columns=factors)
    corr = pivot.corr(min_periods=max(10, len(factors) * 2)).reindex(index=factors, columns=factors)
    corr = corr.fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    shrink = float(params.get("correlation_shrinkage", 0.20))
    shrunk = (1.0 - shrink) * corr.to_numpy(dtype=float) + shrink * np.eye(len(factors))
    try:
        raw = np.linalg.solve(shrunk + 1e-8 * np.eye(len(factors)), mu.to_numpy(dtype=float))
    except np.linalg.LinAlgError:
        raw = mu.to_numpy(dtype=float)
    weights = _normalize_nonnegative_with_cap(pd.Series(raw, index=factors), max_weight)
    return weights, mu


def _pca_weights(
    history: pd.DataFrame,
    factors: list[str],
    params: dict,
    anchor_factor: str,
    max_weight: float,
) -> tuple[pd.Series, pd.Series]:
    pivot = history.pivot_table(index=["date", "ISIN"], columns="factor", values="prediction_score")
    pivot = pivot.reindex(columns=factors).dropna(how="all")
    if len(pivot) < max(30, 5 * len(factors)):
        raise ValueError("Insufficient rows for PCA")
    X = pivot.fillna(0.0).to_numpy(dtype=float)
    model = PCA(n_components=1)
    model.fit(X)
    loadings = pd.Series(model.components_[0], index=factors, dtype=float)
    sign_alignment = str(params.get("sign_alignment", params.get("pca_sign_alignment", "group_average"))).lower()
    if sign_alignment == "anchor_factor" and anchor_factor in loadings.index:
        if loadings.loc[anchor_factor] < 0:
            loadings *= -1.0
    elif loadings.sum() < 0:
        loadings *= -1.0
    weights = _normalize_signed(loadings, max_weight)
    return weights, loadings


def build_group_oof(
    factor_predictions: pd.DataFrame,
    factor_master: pd.DataFrame,
    group_settings: pd.DataFrame,
    method_params: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_col, isin_col = config["columns"]["date"], config["columns"]["isin"]
    fm = factor_master[factor_master["Enabled"].eq(1)].copy()
    gs = group_settings[group_settings["Enabled"].eq(1)].copy()
    f_lookup = factor_lookup(fm)
    g_lookup = group_lookup(gs)
    p_lookup = method_param_lookup(method_params)
    factor_to_group = fm.set_index("Factor_Code")["Factor_Group"].to_dict()

    fp = factor_predictions.copy()
    fp["group"] = fp["factor"].map(factor_to_group)
    fp = fp[fp["group"].isin(set(gs["Factor_Group"]))].copy()
    metric = (
        fp.groupby(["factor", date_col])
        .apply(lambda g: safe_spearman(g["prediction"], g["forward_return"]), include_groups=False)
        .rename("rank_ic")
        .reset_index()
        .rename(columns={date_col: "date"})
    )
    weight_rows: list[dict] = []
    group_rows: list[dict] = []
    previous_weights: dict[str, pd.Series] = {}
    defaults = config["model"].get("group_method_defaults", {})

    for date in sorted(fp[date_col].unique()):
        current = fp[fp[date_col] == date]
        for group, gg in current.groupby("group"):
            settings = g_lookup[group]
            method = str(settings.get("Aggregation_Method", "equal_weight")).lower()
            fallback = str(settings.get("Fallback_Method", "equal_weight")).lower()
            lookback = int(settings.get("Lookback_Periods", 36))
            min_periods = int(settings.get("Min_Periods", 18))
            max_weight = float(settings.get("Max_Weight", 0.50))
            smoothing = float(settings.get("Weight_Smoothing", 0.50))
            anchor = str(settings.get("PCA_Anchor_Factor", ""))
            params = {**defaults, **p_lookup.get(group, {})}
            factors = [f for f in fm.loc[fm["Factor_Group"].eq(group), "Factor_Code"] if f in set(gg["factor"])]
            if not factors:
                continue
            base_weights = pd.Series({f: float(f_lookup[f].get("Base_Weight", 1.0)) for f in factors})
            history_dates = sorted(fp.loc[fp[date_col] < date, date_col].unique())[-lookback:]
            history = fp[(fp[date_col].isin(history_dates)) & (fp["group"].eq(group)) & (fp["factor"].isin(factors))].copy()
            hist_metric = metric[(metric["date"].isin(history_dates)) & (metric["factor"].isin(factors))]
            fallback_used = False
            mean_ic = _factor_ic_means(hist_metric, factors, params)
            raw_loading = pd.Series(np.nan, index=factors, dtype=float)

            try:
                if method == "manual":
                    weights = _normalize_nonnegative_with_cap(base_weights, max_weight)
                elif method == "ic_adjusted":
                    if len(history_dates) < min_periods:
                        raise ValueError("Insufficient IC history")
                    weights, mean_ic = _ic_adjusted_weights(history, hist_metric, factors, params, max_weight)
                elif method == "pca":
                    if len(history_dates) < min_periods or len(factors) < 2:
                        raise ValueError("Insufficient PCA history")
                    weights, raw_loading = _pca_weights(history, factors, params, anchor, max_weight)
                else:
                    weights = pd.Series(1.0 / len(factors), index=factors, dtype=float)
            except (ValueError, np.linalg.LinAlgError):
                weights = _fallback_weights(factors, base_weights, fallback)
                fallback_used = True

            prev = previous_weights.get(group)
            if prev is not None:
                aligned_prev = prev.reindex(factors).fillna(0.0)
                weights = smoothing * aligned_prev + (1.0 - smoothing) * weights
                if method == "pca" and not fallback_used:
                    weights = _normalize_signed(weights, max_weight)
                else:
                    weights = _normalize_nonnegative_with_cap(weights, max_weight)
            previous_weights[group] = weights.copy()

            for factor, value in weights.items():
                meta = f_lookup[factor]
                weight_rows.append({
                    "date": date,
                    "group": group,
                    "group_display_name": settings.get("Display_Name", group),
                    "factor": factor,
                    "factor_name_jp": meta.get("Factor_Name_JP", factor),
                    "factor_name_en": meta.get("Factor_Name_EN", factor),
                    "method": method,
                    "fallback_used": fallback_used,
                    "weight": float(value),
                    "mean_rank_ic": float(mean_ic.get(factor, np.nan)),
                    "pca_raw_loading": float(raw_loading.get(factor, np.nan)),
                })

            pivot = gg.pivot_table(index=[date_col, isin_col], columns="factor", values="prediction_score")
            target_map = gg.groupby([date_col, isin_col])["forward_return"].first()
            available = [f for f in factors if f in pivot.columns]
            if not available:
                continue
            w = weights.reindex(available).to_numpy(dtype=float)
            vals = pivot[available].to_numpy(dtype=float)
            mask = np.isfinite(vals)
            numerator = np.nansum(vals * w, axis=1)
            denominator = np.sum(mask * np.abs(w), axis=1)
            score = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            for keys, value in zip(pivot.index, score):
                group_rows.append({
                    date_col: keys[0],
                    isin_col: keys[1],
                    "forward_return": target_map.get(keys, np.nan),
                    "group": group,
                    "group_display_name": settings.get("Display_Name", group),
                    "aggregation_method": method,
                    "group_prediction": value,
                })

    return pd.DataFrame(group_rows), pd.DataFrame(weight_rows)

def build_composite_oof(group_predictions: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_col, isin_col = config["columns"]["date"], config["columns"]["isin"]
    pivot = group_predictions.pivot_table(
        index=[date_col, isin_col, "forward_return"], columns="group", values="group_prediction"
    ).reset_index()
    feature_cols = [c for c in pivot.columns if c not in [date_col, isin_col, "forward_return"]]
    dates = sorted(pivot[date_col].unique())
    min_train = int(config["model"]["composite_minimum_train_periods"])
    rows: list[pd.DataFrame] = []
    coef_rows: list[dict] = []
    alphas = np.asarray(config["model"]["ridge_alphas"], dtype=float)

    for pos, date in enumerate(dates):
        test = pivot[pivot[date_col] == date].copy()
        if pos < min_train or config["model"]["composite_method"] != "ridge":
            test["stock_alpha"] = test[feature_cols].mean(axis=1)
            test["composite_method"] = "equal_group_mean"
        else:
            train_dates = dates[max(0, pos - int(config["model"]["training_window_periods"])) : pos]
            train = pivot[pivot[date_col].isin(train_dates)].dropna(subset=["forward_return"])
            X_train = train[feature_cols].fillna(0.0)
            X_test = test[feature_cols].fillna(0.0)
            model = RidgeCV(alphas=alphas, fit_intercept=True)
            sw = _date_equal_weights(train, date_col) if config["model"].get("date_equal_weighting", True) else None
            model.fit(X_train, train["forward_return"], sample_weight=sw)
            test["stock_alpha"] = model.predict(X_test)
            test["composite_method"] = "ridge"
            coef_rows.append({
                "test_date": date,
                "intercept": float(model.intercept_),
                "ridge_alpha": float(model.alpha_),
                **{f"coef_{c}": float(v) for c, v in zip(feature_cols, model.coef_)},
            })
        test["stock_score"] = 2.0 * rank_to_unit_interval(test["stock_alpha"]) - 1.0
        test["prediction_dispersion"] = test[feature_cols].std(axis=1)
        test["confidence_score"] = test["stock_alpha"] / (test["prediction_dispersion"].fillna(0.0) + 1e-6)
        rows.append(test)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coef_rows)


def fit_latest_stock_scores(
    panel: pd.DataFrame,
    factor_map: dict[str, str],
    run: ModelRunResult,
    factor_oof: pd.DataFrame,
    group_oof: pd.DataFrame,
    factor_master: pd.DataFrame,
    group_settings: pd.DataFrame,
    method_params: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = config["columns"]
    date_col, isin_col = cols["date"], cols["isin"]
    latest_date = panel[date_col].max()
    history = panel[panel["forward_return"].notna()]
    latest = panel[panel[date_col] == latest_date].copy()
    selected = run.selection[run.selection["adopted"]].set_index("factor")
    factor_rows: list[pd.DataFrame] = []

    for factor, sel in selected.iterrows():
        z_col = factor_map[factor]
        train = history[[date_col, isin_col, z_col, "forward_return"]].dropna()
        if len(train) < 50:
            continue
        sw = _date_equal_weights(train, date_col) if config["model"].get("date_equal_weighting", True) else None
        model_name = sel["selected_model"]
        model = _fit(model_name, train[z_col].to_numpy(), train["forward_return"].to_numpy(), sw, config)
        block = latest[[date_col, isin_col, z_col]].copy()
        block["factor"] = factor
        block["prediction"] = _predict(model, model_name, block[z_col].to_numpy(), config)
        block["prediction_score"] = 2.0 * rank_to_unit_interval(block["prediction"]) - 1.0
        factor_rows.append(block[[date_col, isin_col, "factor", "prediction", "prediction_score"]])
    latest_factor = pd.concat(factor_rows, ignore_index=True)

    # Reuse group construction logic by adding placeholder target and historical IC-derived weights.
    latest_factor["forward_return"] = np.nan
    combined_factor = pd.concat([factor_oof, latest_factor], ignore_index=True, sort=False)
    combined_group, group_weights = build_group_oof(
        combined_factor, factor_master, group_settings, method_params, config
    )
    latest_group = combined_group[combined_group[date_col] == latest_date].copy()

    pivot_latest = latest_group.pivot_table(index=[date_col, isin_col], columns="group", values="group_prediction").reset_index()
    feature_cols = [c for c in pivot_latest.columns if c not in [date_col, isin_col]]
    if config["model"]["composite_method"] == "ridge":
        hist_pivot = group_oof.pivot_table(
            index=[date_col, isin_col, "forward_return"], columns="group", values="group_prediction"
        ).reset_index()
        common = [c for c in feature_cols if c in hist_pivot.columns]
        if common and len(hist_pivot.dropna(subset=["forward_return"])) > 50:
            model = RidgeCV(alphas=np.asarray(config["model"]["ridge_alphas"], dtype=float), fit_intercept=True)
            hist_train = hist_pivot.dropna(subset=["forward_return"])
            model.fit(hist_train[common].fillna(0.0), hist_train["forward_return"])
            pivot_latest["stock_alpha"] = model.predict(pivot_latest[common].fillna(0.0))
        else:
            pivot_latest["stock_alpha"] = pivot_latest[feature_cols].mean(axis=1)
    else:
        pivot_latest["stock_alpha"] = pivot_latest[feature_cols].mean(axis=1)
    pivot_latest["stock_score"] = 2.0 * rank_to_unit_interval(pivot_latest["stock_alpha"]) - 1.0
    pivot_latest["prediction_dispersion"] = pivot_latest[feature_cols].std(axis=1)
    pivot_latest["confidence_score"] = pivot_latest["stock_alpha"] / (pivot_latest["prediction_dispersion"].fillna(0.0) + 1e-6)
    latest_group_weights = group_weights[group_weights["date"].eq(latest_date)].copy() if not group_weights.empty else group_weights
    return pivot_latest, latest_factor, latest_group, latest_group_weights
