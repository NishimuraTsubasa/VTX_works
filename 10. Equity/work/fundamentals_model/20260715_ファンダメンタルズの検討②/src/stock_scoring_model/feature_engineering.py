from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import _transform_raw


@dataclass
class FeatureEngineeringResult:
    panel: pd.DataFrame
    factor_master: pd.DataFrame
    factor_codes: list[str]
    lineage: pd.DataFrame


def _resolve_control(
    factor_code: str,
    factor_group: str,
    control: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    defaults = config.get("feature_engineering", {}).get("defaults", {})
    resolved = {
        "Enabled": int(bool(defaults.get("enabled", False))),
        "Generation_Mode": str(defaults.get("generation_mode", "selected")).lower(),
        "Include_Raw": int(bool(defaults.get("include_raw", True))),
        "Scope_Type": "default",
        "Scope_Value": "DEFAULT",
    }
    if control is None or control.empty:
        return resolved
    factor_match = control[(control["Scope_Type"].eq("factor")) & (control["Scope_Value"].eq(factor_code))]
    group_match = control[(control["Scope_Type"].eq("group")) & (control["Scope_Value"].eq(factor_group))]
    match = factor_match if not factor_match.empty else group_match
    if match.empty:
        return resolved
    row = match.iloc[-1]
    for col in ["Enabled", "Generation_Mode", "Include_Raw", "Scope_Type", "Scope_Value"]:
        resolved[col] = row[col]
    resolved["Enabled"] = int(resolved["Enabled"])
    resolved["Include_Raw"] = int(resolved["Include_Raw"])
    resolved["Generation_Mode"] = str(resolved["Generation_Mode"]).lower()
    return resolved


def _rules_for_factor(
    factor_code: str,
    factor_group: str,
    rules: pd.DataFrame,
    generation_mode: str,
) -> pd.DataFrame:
    if rules is None or rules.empty:
        return pd.DataFrame(columns=rules.columns if rules is not None else [])
    matched = rules[
        ((rules["Scope_Type"].eq("factor")) & (rules["Scope_Value"].eq(factor_code)))
        | ((rules["Scope_Type"].eq("group")) & (rules["Scope_Value"].eq(factor_group)))
    ].copy()
    matched = matched[matched["Enabled"].astype(int).eq(1)]
    if str(generation_mode).lower() == "selected":
        matched = matched[matched["Selected"].astype(int).eq(1)]
    return matched.drop_duplicates("Rule_ID", keep="last")


def _direction(base_direction: int, row: pd.Series) -> int:
    mode = str(row.get("Direction_Mode", "inherit")).lower()
    if mode == "reverse":
        return -int(base_direction)
    if mode == "custom":
        custom = int(row.get("Custom_Direction", 1))
        return custom if custom in {-1, 1} else int(base_direction)
    return int(base_direction)


def _feature_code(base: str, row: pd.Series) -> str:
    feature_type = str(row["Feature_Type"]).lower()
    lag = int(row.get("Source_Lag_Periods", 1))
    if feature_type == "difference":
        p = int(row.get("Difference_Periods", 1))
        return f"{base}__DIFF_P{p}_L{lag}"
    if feature_type == "rolling_mean_deviation":
        w = int(row.get("Window_Periods", 12))
        return f"{base}__MADEV_W{w}_L{lag}"
    if feature_type == "rolling_mean_ratio":
        w = int(row.get("Window_Periods", 12))
        return f"{base}__MARATIO_W{w}_L{lag}"
    if feature_type == "expanding_mean_deviation":
        return f"{base}__EXPDEV_L{lag}"
    raise ValueError(f"Unsupported derived feature type: {feature_type}")


def _feature_names(base_jp: str, base_en: str, row: pd.Series) -> tuple[str, str]:
    feature_type = str(row["Feature_Type"]).lower()
    lag = int(row.get("Source_Lag_Periods", 1))
    if feature_type == "difference":
        p = int(row.get("Difference_Periods", 1))
        return f"{base_jp} 差分({p}期, 情報ラグ{lag})", f"{base_en} Difference({p}, lag {lag})"
    if feature_type == "rolling_mean_deviation":
        w = int(row.get("Window_Periods", 12))
        return f"{base_jp} 移動平均乖離({w}期, 情報ラグ{lag})", f"{base_en} Rolling Mean Deviation({w}, lag {lag})"
    if feature_type == "rolling_mean_ratio":
        w = int(row.get("Window_Periods", 12))
        return f"{base_jp} 移動平均比率({w}期, 情報ラグ{lag})", f"{base_en} Rolling Mean Ratio({w}, lag {lag})"
    if feature_type == "expanding_mean_deviation":
        return f"{base_jp} 過去平均乖離(情報ラグ{lag})", f"{base_en} Expanding Mean Deviation(lag {lag})"
    return base_jp, base_en


def _derive_for_one_stock(values: pd.Series, row: pd.Series) -> pd.Series:
    feature_type = str(row["Feature_Type"]).lower()
    source_lag = int(row.get("Source_Lag_Periods", 1))
    source = values.shift(source_lag)
    if feature_type == "difference":
        periods = int(row.get("Difference_Periods", 1))
        return source - values.shift(source_lag + periods)

    exclude_source = bool(int(row.get("Exclude_Source_From_Baseline", 1)))
    baseline_shift = source_lag + (1 if exclude_source else 0)
    baseline_source = values.shift(baseline_shift)
    min_periods = max(1, int(row.get("Min_Periods", 3)))

    if feature_type in {"rolling_mean_deviation", "rolling_mean_ratio"}:
        window = max(1, int(row.get("Window_Periods", 12)))
        baseline = baseline_source.rolling(window=window, min_periods=min(min_periods, window)).mean()
        if feature_type == "rolling_mean_deviation":
            return source - baseline
        return source / baseline.where(baseline.abs() > 1e-12) - 1.0

    if feature_type == "expanding_mean_deviation":
        baseline = baseline_source.expanding(min_periods=min_periods).mean()
        return source - baseline

    raise ValueError(f"Unsupported derived feature type: {feature_type}")


def build_engineered_factor_panel(
    stocks: pd.DataFrame,
    base_factor_codes: list[str],
    factor_master: pd.DataFrame,
    feature_control: pd.DataFrame,
    derived_rules: pd.DataFrame,
    config: dict[str, Any],
) -> FeatureEngineeringResult:
    """Create lag-safe derived factors and an expanded factor master.

    Derived values are stored on scoring date t, but use source observations no later than
    t - Source_Lag_Periods. With the default one-period forward target, a source lag of 1
    creates a two-period gap between the latest source factor observation (t-1) and the
    realized target return (t+1).
    """
    cols = config["columns"]
    date_col, isin_col = cols["date"], cols["isin"]
    base_horizon = int(config.get("target", {}).get("stock_horizon_periods", 1))
    enabled_global = bool(config.get("feature_engineering", {}).get("enabled", True))

    out = stocks.sort_values([isin_col, date_col]).copy()
    base_master = factor_master[factor_master["Factor_Code"].astype(str).isin(base_factor_codes)].copy()
    expanded_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    all_codes: list[str] = []

    for _, base_row in base_master.iterrows():
        base = str(base_row["Factor_Code"])
        group = str(base_row["Factor_Group"])
        control = _resolve_control(base, group, feature_control, config)
        include_raw = bool(control["Include_Raw"]) or not (enabled_global and bool(control["Enabled"]))
        # Base factors are always preprocessed so the S00-S05 baseline scenarios remain available.
        # Include_Raw controls whether the raw series enters the enhanced S06/S07 model.
        raw_row = base_row.to_dict()
        raw_row.update({
            "Enabled": int(base_row.get("Enabled", 1)) if include_raw else 0,
            "Base_Factor_Code": base,
            "Feature_Type": "raw",
            "Rule_ID": "RAW",
            "Source_Lag_Periods": 0,
            "Effective_Target_Gap_Periods": base_horizon,
            "Is_Derived": 0,
        })
        expanded_rows.append(raw_row)
        all_codes.append(base)
        lineage_rows.append({
            "Factor_Code": base,
            "Base_Factor_Code": base,
            "Factor_Group": group,
            "Feature_Type": "raw",
            "Rule_ID": "RAW",
            "Source_Lag_Periods": 0,
            "Base_Target_Horizon_Periods": base_horizon,
            "Effective_Target_Gap_Periods": base_horizon,
            "Formula": "x[t]",
            "Control_Scope": control["Scope_Type"],
            "Control_Value": control["Scope_Value"],
            "Generation_Mode": control["Generation_Mode"],
            "Used_In_Model": int(include_raw),
        })

        if not enabled_global or not bool(control["Enabled"]):
            continue
        applicable = _rules_for_factor(base, group, derived_rules, control["Generation_Mode"])
        if applicable.empty:
            continue

        transformed = _transform_raw(out[base], str(base_row.get("Transform", "none")).lower())
        temp = pd.DataFrame({isin_col: out[isin_col], date_col: out[date_col], "value": transformed}, index=out.index)
        for _, rule in applicable.iterrows():
            code = _feature_code(base, rule)
            if code in out.columns:
                continue
            derived = temp.groupby(isin_col, group_keys=False)["value"].apply(lambda s: _derive_for_one_stock(s, rule))
            derived = derived.reindex(out.index)
            out[code] = derived

            jp, en = _feature_names(str(base_row["Factor_Name_JP"]), str(base_row["Factor_Name_EN"]), rule)
            drow = base_row.to_dict()
            drow.update({
                "Factor_Code": code,
                "Factor_Name_JP": jp,
                "Factor_Name_EN": en,
                "Direction": _direction(int(base_row["Direction"]), rule),
                "Transform": "none",
                "Base_Factor_Code": base,
                "Feature_Type": str(rule["Feature_Type"]),
                "Rule_ID": str(rule["Rule_ID"]),
                "Source_Lag_Periods": int(rule["Source_Lag_Periods"]),
                "Effective_Target_Gap_Periods": base_horizon + int(rule["Source_Lag_Periods"]),
                "Is_Derived": 1,
                "Description": str(rule.get("Description", "")) or jp,
            })
            expanded_rows.append(drow)
            all_codes.append(code)

            ft = str(rule["Feature_Type"])
            source_lag = int(rule["Source_Lag_Periods"])
            if ft == "difference":
                p = int(rule["Difference_Periods"])
                formula = f"x[t-{source_lag}] - x[t-{source_lag + p}]"
            elif ft == "rolling_mean_deviation":
                w = int(rule["Window_Periods"])
                formula = f"x[t-{source_lag}] - mean(prior {w} values, source excluded={int(rule['Exclude_Source_From_Baseline'])})"
            elif ft == "rolling_mean_ratio":
                w = int(rule["Window_Periods"])
                formula = f"x[t-{source_lag}] / mean(prior {w} values) - 1"
            else:
                formula = f"x[t-{source_lag}] - expanding_mean(prior values)"
            lineage_rows.append({
                "Factor_Code": code,
                "Base_Factor_Code": base,
                "Factor_Group": group,
                "Feature_Type": ft,
                "Rule_ID": str(rule["Rule_ID"]),
                "Difference_Periods": int(rule["Difference_Periods"]),
                "Window_Periods": int(rule["Window_Periods"]),
                "Min_Periods": int(rule["Min_Periods"]),
                "Source_Lag_Periods": source_lag,
                "Base_Target_Horizon_Periods": base_horizon,
                "Effective_Target_Gap_Periods": base_horizon + source_lag,
                "Formula": formula,
                "Control_Scope": control["Scope_Type"],
                "Control_Value": control["Scope_Value"],
                "Generation_Mode": control["Generation_Mode"],
                "Selected_Rule": int(rule["Selected"]),
                "Used_In_Model": 1,
            })

    expanded = pd.DataFrame(expanded_rows)
    if expanded.empty:
        expanded = base_master.copy()
    # Preserve the original master column order and append lineage columns.
    original_cols = list(factor_master.columns)
    extra_cols = [c for c in expanded.columns if c not in original_cols]
    expanded = expanded[original_cols + extra_cols].reset_index(drop=True)
    lineage = pd.DataFrame(lineage_rows)
    return FeatureEngineeringResult(out, expanded, all_codes, lineage)
