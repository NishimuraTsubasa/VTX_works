from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FACTOR_REQUIRED_COLUMNS = [
    "Factor_Code",
    "Factor_Name_JP",
    "Factor_Name_EN",
    "Factor_Group",
    "Enabled",
    "Direction",
    "Base_Weight",
    "Transform",
    "Winsorize",
    "Neutralize",
    "Rank_Normalize",
    "Min_Coverage",
]

GROUP_REQUIRED_COLUMNS = [
    "Factor_Group",
    "Display_Name",
    "Enabled",
    "Aggregation_Method",
    "Lookback_Periods",
    "Min_Periods",
    "Max_Weight",
    "Weight_Smoothing",
    "Fallback_Method",
]

ALLOWED_METHODS = {"equal_weight", "manual", "ic_adjusted", "pca"}
ALLOWED_FALLBACKS = {"equal_weight", "manual"}
ALLOWED_TRANSFORMS = {"none", "log", "log1p", "inverse", "signed_log"}
ALLOWED_WINSOR = {"default", "none", "1_99", "2.5_97.5", "mad_3"}


@dataclass
class FactorSettingsBundle:
    factor_master: pd.DataFrame
    group_settings: pd.DataFrame
    method_params: pd.DataFrame
    validation: pd.DataFrame


def _coerce_bool01(s: pd.Series, default: int = 1) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(int).clip(0, 1)


def _clean_factor_master(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(how="all")
    out.columns = [str(c).strip() for c in out.columns]
    for col in FACTOR_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["Factor_Code"] = out["Factor_Code"].astype(str).str.strip()
    out = out[out["Factor_Code"].ne("") & out["Factor_Code"].ne("nan")].copy()
    out["Factor_Name_JP"] = out["Factor_Name_JP"].fillna(out["Factor_Code"]).astype(str).str.strip()
    out["Factor_Name_EN"] = out["Factor_Name_EN"].fillna(out["Factor_Code"]).astype(str).str.strip()
    out["Factor_Group"] = out["Factor_Group"].astype(str).str.strip()
    out["Enabled"] = _coerce_bool01(out["Enabled"], 1)
    out["Direction"] = pd.to_numeric(out["Direction"], errors="coerce").fillna(1).astype(int)
    out["Base_Weight"] = pd.to_numeric(out["Base_Weight"], errors="coerce").fillna(1.0)
    out["Transform"] = out["Transform"].fillna("none").astype(str).str.strip().str.lower()
    out["Winsorize"] = out["Winsorize"].fillna("default").astype(str).str.strip().str.lower()
    out["Neutralize"] = _coerce_bool01(out["Neutralize"], 1)
    out["Rank_Normalize"] = _coerce_bool01(out["Rank_Normalize"], 1)
    out["Min_Coverage"] = pd.to_numeric(out["Min_Coverage"], errors="coerce").fillna(0.60).clip(0, 1)
    if "Description" not in out:
        out["Description"] = ""
    out["Description"] = out["Description"].fillna("").astype(str)
    return out.reset_index(drop=True)


def _clean_group_settings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().dropna(how="all")
    out.columns = [str(c).strip() for c in out.columns]
    for col in GROUP_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["Factor_Group"] = out["Factor_Group"].astype(str).str.strip()
    out = out[out["Factor_Group"].ne("") & out["Factor_Group"].ne("nan")].copy()
    out["Display_Name"] = out["Display_Name"].fillna(out["Factor_Group"]).astype(str).str.strip()
    out["Enabled"] = _coerce_bool01(out["Enabled"], 1)
    out["Aggregation_Method"] = out["Aggregation_Method"].fillna("equal_weight").astype(str).str.strip().str.lower()
    out["Lookback_Periods"] = pd.to_numeric(out["Lookback_Periods"], errors="coerce").fillna(36).astype(int)
    out["Min_Periods"] = pd.to_numeric(out["Min_Periods"], errors="coerce").fillna(18).astype(int)
    out["Max_Weight"] = pd.to_numeric(out["Max_Weight"], errors="coerce").fillna(0.50).clip(0, 1)
    out["Weight_Smoothing"] = pd.to_numeric(out["Weight_Smoothing"], errors="coerce").fillna(0.50).clip(0, 1)
    out["Fallback_Method"] = out["Fallback_Method"].fillna("equal_weight").astype(str).str.strip().str.lower()
    if "PCA_Anchor_Factor" not in out:
        out["PCA_Anchor_Factor"] = ""
    if "Notes" not in out:
        out["Notes"] = ""
    out["PCA_Anchor_Factor"] = out["PCA_Anchor_Factor"].fillna("").astype(str).str.strip()
    out["Notes"] = out["Notes"].fillna("").astype(str)
    return out.reset_index(drop=True)


def _clean_method_params(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Factor_Group", "Param_Name", "Param_Value", "Description"])
    out = df.copy().dropna(how="all")
    out.columns = [str(c).strip() for c in out.columns]
    for col in ["Factor_Group", "Param_Name", "Param_Value"]:
        if col not in out.columns:
            out[col] = np.nan
    if "Description" not in out:
        out["Description"] = ""
    out = out.dropna(subset=["Factor_Group", "Param_Name"])
    out["Factor_Group"] = out["Factor_Group"].astype(str).str.strip()
    out["Param_Name"] = out["Param_Name"].astype(str).str.strip()
    return out.reset_index(drop=True)


def validate_factor_settings(
    factor_master: pd.DataFrame,
    group_settings: pd.DataFrame,
    stock_columns: list[str] | None = None,
) -> pd.DataFrame:
    issues: list[dict[str, Any]] = []

    duplicated = factor_master[factor_master["Factor_Code"].duplicated(keep=False)]
    for code in sorted(duplicated["Factor_Code"].unique()):
        issues.append({"severity": "ERROR", "check": "duplicate_factor_code", "item": code, "detail": "Factor_Code is duplicated."})

    invalid_direction = factor_master[~factor_master["Direction"].isin([-1, 1])]
    for _, row in invalid_direction.iterrows():
        issues.append({"severity": "ERROR", "check": "invalid_direction", "item": row["Factor_Code"], "detail": f"Direction={row['Direction']} must be -1 or 1."})

    invalid_transform = factor_master[~factor_master["Transform"].isin(ALLOWED_TRANSFORMS)]
    for _, row in invalid_transform.iterrows():
        issues.append({"severity": "ERROR", "check": "invalid_transform", "item": row["Factor_Code"], "detail": f"Transform={row['Transform']}"})

    invalid_winsor = factor_master[~factor_master["Winsorize"].isin(ALLOWED_WINSOR)]
    for _, row in invalid_winsor.iterrows():
        issues.append({"severity": "ERROR", "check": "invalid_winsorize", "item": row["Factor_Code"], "detail": f"Winsorize={row['Winsorize']}"})

    duplicated_groups = group_settings[group_settings["Factor_Group"].duplicated(keep=False)]
    for group in sorted(duplicated_groups["Factor_Group"].unique()):
        issues.append({"severity": "ERROR", "check": "duplicate_group", "item": group, "detail": "Factor_Group is duplicated in Group_Settings."})

    invalid_methods = group_settings[~group_settings["Aggregation_Method"].isin(ALLOWED_METHODS)]
    for _, row in invalid_methods.iterrows():
        issues.append({"severity": "ERROR", "check": "invalid_aggregation_method", "item": row["Factor_Group"], "detail": f"Aggregation_Method={row['Aggregation_Method']}"})

    invalid_fallbacks = group_settings[~group_settings["Fallback_Method"].isin(ALLOWED_FALLBACKS)]
    for _, row in invalid_fallbacks.iterrows():
        issues.append({"severity": "ERROR", "check": "invalid_fallback_method", "item": row["Factor_Group"], "detail": f"Fallback_Method={row['Fallback_Method']}"})

    known_groups = set(group_settings["Factor_Group"])
    for group in sorted(set(factor_master["Factor_Group"]) - known_groups):
        issues.append({"severity": "ERROR", "check": "group_not_defined", "item": group, "detail": "Factor_Group is not present in Group_Settings."})

    if stock_columns is not None:
        active = factor_master[factor_master["Enabled"].eq(1)]
        missing = sorted(set(active["Factor_Code"]) - set(stock_columns))
        for code in missing:
            issues.append({"severity": "ERROR", "check": "configured_factor_missing", "item": code, "detail": "Enabled factor is absent from stock factor input."})
        unknown = sorted({c for c in stock_columns if str(c).startswith("FA")} - set(factor_master["Factor_Code"]))
        for code in unknown:
            issues.append({"severity": "WARNING", "check": "input_factor_not_configured", "item": code, "detail": "Input factor column is not defined in Factor_Master and will be ignored."})

    if not issues:
        issues.append({"severity": "OK", "check": "factor_master_validation", "item": "ALL", "detail": "No validation issues were found."})
    return pd.DataFrame(issues)


def load_factor_settings(path: str | Path, sheet_map: dict[str, str] | None = None, stock_columns: list[str] | None = None) -> FactorSettingsBundle:
    path = Path(path)
    sheet_map = sheet_map or {}
    factor_sheet = sheet_map.get("factor_master", "Factor_Master")
    group_sheet = sheet_map.get("group_settings", "Group_Settings")
    param_sheet = sheet_map.get("method_params", "Group_Method_Params")
    xls = pd.ExcelFile(path)
    factor = _clean_factor_master(pd.read_excel(path, sheet_name=factor_sheet))
    groups = _clean_group_settings(pd.read_excel(path, sheet_name=group_sheet))
    params = _clean_method_params(pd.read_excel(path, sheet_name=param_sheet) if param_sheet in xls.sheet_names else None)
    validation = validate_factor_settings(factor, groups, stock_columns)
    errors = validation[validation["severity"].eq("ERROR")]
    if not errors.empty:
        details = "; ".join(errors["check"].astype(str) + ":" + errors["item"].astype(str))
        raise ValueError(f"Factor master validation failed: {details}")
    return FactorSettingsBundle(factor, groups, params, validation)


def factor_lookup(factor_master: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return factor_master.set_index("Factor_Code").to_dict(orient="index")


def group_lookup(group_settings: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return group_settings.set_index("Factor_Group").to_dict(orient="index")


def method_param_lookup(method_params: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for _, row in method_params.iterrows():
        group = str(row["Factor_Group"])
        value: Any = row["Param_Value"]
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "yes"}:
                value = True
            elif low in {"false", "no"}:
                value = False
            else:
                try:
                    value = float(value) if "." in value else int(value)
                except ValueError:
                    pass
        out.setdefault(group, {})[str(row["Param_Name"])] = value
    return out


def active_factor_codes(factor_master: pd.DataFrame, group_settings: pd.DataFrame) -> list[str]:
    enabled_groups = set(group_settings.loc[group_settings["Enabled"].eq(1), "Factor_Group"])
    active = factor_master[
        factor_master["Enabled"].eq(1) & factor_master["Factor_Group"].isin(enabled_groups)
    ]
    return active["Factor_Code"].tolist()


def resolved_factor_settings(factor_master: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    default_w = config["preprocessing"]["winsorize"]
    default_neu = config["preprocessing"]["neutralization"].get("enabled", True)
    default_rank = config["preprocessing"].get("rank_transform", "gaussian")
    for _, row in factor_master.iterrows():
        winsor = row["Winsorize"]
        if winsor == "default":
            winsor = (
                f"{100*default_w['lower_quantile']:.1f}_{100*default_w['upper_quantile']:.1f}"
                if default_w.get("enabled", True) else "none"
            )
        rows.append({
            **row.to_dict(),
            "Winsorize_Resolved": winsor,
            "Neutralize_Resolved": int(row["Neutralize"]) if pd.notna(row["Neutralize"]) else int(default_neu),
            "Rank_Normalize_Resolved": int(row["Rank_Normalize"]),
            "Rank_Transform_Default": default_rank,
            "Status": "Used" if int(row["Enabled"]) == 1 else "Disabled",
        })
    return pd.DataFrame(rows)
