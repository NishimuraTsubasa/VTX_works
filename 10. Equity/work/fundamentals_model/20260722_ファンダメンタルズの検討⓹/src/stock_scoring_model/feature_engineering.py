from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .master import FactorMeta


def add_forward_return(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    c = config["columns"]
    out = data.copy()
    out[c["date"]] = pd.to_datetime(out[c["date"]])
    out = out.sort_values([c["isin"], c["date"]]).reset_index(drop=True)
    horizon = int(config["target"].get("stock_horizon_periods", 1))
    alignment = config["target"].get("stock_return_alignment", "contemporaneous_to_forward")
    if alignment == "contemporaneous_to_forward":
        out["NextMonthReturn"] = out.groupby(c["isin"], sort=False)[c["stock_return"]].shift(-horizon)
    elif alignment == "already_forward":
        out["NextMonthReturn"] = pd.to_numeric(out[c["stock_return"]], errors="coerce")
    else:
        raise ValueError(f"未対応のstock_return_alignmentです: {alignment}")
    return out


def _control_for(code: str, group: str, control: pd.DataFrame) -> tuple[bool, str, bool]:
    if control.empty:
        return False, "selected", True
    ctl = control.copy()
    ctl = ctl[ctl.get("Scope_Value").notna()]
    factor_rows = ctl[(ctl["Scope_Type"].astype(str).str.lower() == "factor") & (ctl["Scope_Value"].astype(str) == code)]
    group_rows = ctl[(ctl["Scope_Type"].astype(str).str.lower() == "group") & (ctl["Scope_Value"].astype(str) == group)]
    row = factor_rows.iloc[0] if not factor_rows.empty else (group_rows.iloc[0] if not group_rows.empty else None)
    if row is None:
        return False, "selected", True
    return bool(int(row.get("Enabled", 0))), str(row.get("Generation_Mode", "selected")).lower(), bool(int(row.get("Include_Raw", 1)))


def _applicable_rules(code: str, group: str, rules: pd.DataFrame, mode: str) -> pd.DataFrame:
    if rules.empty:
        return rules
    r = rules.copy()
    r = r[r.get("Scope_Value").notna()]
    mask = (
        ((r["Scope_Type"].astype(str).str.lower() == "factor") & (r["Scope_Value"].astype(str) == code))
        | ((r["Scope_Type"].astype(str).str.lower() == "group") & (r["Scope_Value"].astype(str) == group))
    )
    r = r[mask]
    r = r[pd.to_numeric(r.get("Enabled", 0), errors="coerce").fillna(0).astype(int) == 1]
    if mode == "selected" and "Selected" in r:
        r = r[pd.to_numeric(r["Selected"], errors="coerce").fillna(0).astype(int) == 1]
    return r


def _direction_from_rule(base: int, row: pd.Series) -> int:
    mode = str(row.get("Direction_Mode", "inherit")).lower()
    if mode == "reverse":
        return -base
    if mode == "custom":
        value = int(pd.to_numeric(row.get("Custom_Direction", base), errors="coerce"))
        return 1 if value >= 0 else -1
    return base


def generate_derived_features(
    data: pd.DataFrame,
    config: dict[str, Any],
    metas: dict[str, FactorMeta],
    control: pd.DataFrame,
    rules: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, FactorMeta], pd.DataFrame]:
    c = config["columns"]
    out = data.sort_values([c["isin"], c["date"]]).copy()
    all_metas = dict(metas)
    lineage: list[dict[str, Any]] = []

    for code, meta in metas.items():
        enabled, mode, include_raw = _control_for(code, meta.group, control)
        if not include_raw:
            all_metas.pop(code, None)
        if not enabled:
            continue
        applicable = _applicable_rules(code, meta.group, rules, mode)
        if applicable.empty:
            continue
        grouped = out.groupby(c["isin"], sort=False)[code]
        for _, row in applicable.iterrows():
            ftype = str(row.get("Feature_Type", "")).lower()
            lag = int(pd.to_numeric(row.get("Source_Lag_Periods", 1), errors="coerce") or 1)
            source = grouped.shift(lag)
            rule_id = str(row.get("Rule_ID", ftype)).strip()
            direction = _direction_from_rule(meta.direction, row)
            if ftype == "difference":
                period = int(pd.to_numeric(row.get("Difference_Periods", 1), errors="coerce") or 1)
                name = f"{code}__DIFF_P{period}_L{lag}"
                out[name] = source - grouped.shift(lag + period)
            elif ftype in {"rolling_mean_deviation", "rolling_mean_ratio"}:
                window = int(pd.to_numeric(row.get("Window_Periods", 12), errors="coerce") or 12)
                minp = int(pd.to_numeric(row.get("Min_Periods", window), errors="coerce") or window)
                exclude = bool(int(pd.to_numeric(row.get("Exclude_Source_From_Baseline", 1), errors="coerce") or 0))
                baseline_lag = lag + 1 if exclude else lag
                baseline_source = grouped.shift(baseline_lag)
                baseline = baseline_source.groupby(out[c["isin"]], sort=False).transform(
                    lambda s: s.rolling(window, min_periods=minp).mean()
                )
                if ftype == "rolling_mean_deviation":
                    name = f"{code}__MADEV_W{window}_L{lag}"
                    out[name] = source - baseline
                else:
                    name = f"{code}__MARATIO_W{window}_L{lag}"
                    out[name] = source / baseline.replace(0, np.nan) - 1.0
            elif ftype == "expanding_mean_deviation":
                minp = int(pd.to_numeric(row.get("Min_Periods", 12), errors="coerce") or 12)
                exclude = bool(int(pd.to_numeric(row.get("Exclude_Source_From_Baseline", 1), errors="coerce") or 0))
                baseline_lag = lag + 1 if exclude else lag
                baseline_source = grouped.shift(baseline_lag)
                baseline = baseline_source.groupby(out[c["isin"]], sort=False).transform(
                    lambda s: s.expanding(min_periods=minp).mean()
                )
                name = f"{code}__EXPDEV_L{lag}"
                out[name] = source - baseline
            else:
                continue
            all_metas[name] = replace(meta, code=name, direction=direction)
            lineage.append({
                "Feature_Code": name,
                "Source_Factor": code,
                "Factor_Group": meta.group,
                "Rule_ID": rule_id,
                "Feature_Type": ftype,
                "Source_Lag_Periods": lag,
                "Target_Horizon_Periods": int(config["target"].get("stock_horizon_periods", 1)),
                "Effective_Source_to_Target_Gap": lag + int(config["target"].get("stock_horizon_periods", 1)),
                "Direction": direction,
            })
    return out, all_metas, pd.DataFrame(lineage)
