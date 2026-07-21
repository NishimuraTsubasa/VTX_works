from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FactorMeta:
    code: str
    group: str
    direction: int
    base_weight: float


def _clean_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy().dropna(how="all")
    out.columns = [str(c).strip() for c in out.columns]
    known_headers = {
        "Factor_Code", "Factor_Group", "Scope_Type", "Rule_ID",
        "Aggregation_Method", "Transform", "Country", "Sector",
        "Sector_Group", "Setting",
    }
    if not known_headers.intersection(out.columns):
        for pos in range(min(12, len(out))):
            row_values = [str(x).strip() for x in out.iloc[pos].tolist() if pd.notna(x)]
            if known_headers.intersection(row_values):
                headers = [str(x).strip() if pd.notna(x) else f"Unnamed_{j}" for j, x in enumerate(out.iloc[pos].tolist())]
                out = out.iloc[pos + 1:].copy()
                out.columns = headers
                break
    return out.dropna(how="all")


def _enabled_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Enabled" in out.columns:
        out["Enabled"] = pd.to_numeric(out["Enabled"], errors="coerce").fillna(0).astype(int)
        out = out[out["Enabled"].eq(1)]
    return out


def parse_master(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    fm = _clean_frame(sheets.get("Factor_Master"))
    gs = _clean_frame(sheets.get("Group_Settings"))
    fc = _clean_frame(sheets.get("Feature_Engineering_Control"))
    dr = _clean_frame(sheets.get("Derived_Feature_Rules"))
    fo = _clean_frame(sheets.get("Factor_Overrides"))
    go = _clean_frame(sheets.get("Group_Overrides"))
    country_region = _clean_frame(sheets.get("Country_Region_Map"))
    sector_group = _clean_frame(sheets.get("Sector_Group_Map"))
    interaction = _clean_frame(sheets.get("Sector_Factor_Interaction"))
    layer3_settings = _clean_frame(sheets.get("Layer3_Settings"))

    required = {"Factor_Code", "Factor_Group", "Enabled", "Direction", "Base_Weight"}
    missing = required - set(fm.columns)
    if missing:
        raise ValueError(f"Factor_Masterの必須列が不足しています: {sorted(missing)}")
    fm = fm[fm["Factor_Code"].notna()].copy()
    fm["Factor_Code"] = fm["Factor_Code"].astype(str).str.strip()
    if fm["Factor_Code"].duplicated().any():
        raise ValueError("Factor_Codeが重複しています。")
    fm["Enabled"] = pd.to_numeric(fm["Enabled"], errors="coerce").fillna(0).astype(int)
    fm["Direction"] = pd.to_numeric(fm["Direction"], errors="coerce").fillna(1).astype(int)
    fm["Base_Weight"] = pd.to_numeric(fm["Base_Weight"], errors="coerce").fillna(1.0)
    if not fm["Direction"].isin([-1, 1]).all():
        raise ValueError("Directionは1または-1のみです。")

    if not gs.empty:
        gs = gs[gs["Factor_Group"].notna()].copy()
        gs["Factor_Group"] = gs["Factor_Group"].astype(str).str.strip()
        gs["Enabled"] = pd.to_numeric(gs["Enabled"], errors="coerce").fillna(0).astype(int)
    enabled_groups = set(gs.loc[gs.get("Enabled", 1).eq(1), "Factor_Group"]) if not gs.empty else set(fm["Factor_Group"])
    enabled = fm[(fm["Enabled"] == 1) & fm["Factor_Group"].isin(enabled_groups)].copy()
    metas = {
        row.Factor_Code: FactorMeta(row.Factor_Code, str(row.Factor_Group), int(row.Direction), float(row.Base_Weight))
        for row in enabled.itertuples(index=False)
    }
    group_methods: dict[str, str] = {}
    if not gs.empty:
        for row in gs.itertuples(index=False):
            if int(getattr(row, "Enabled", 1)) == 1:
                group_methods[str(row.Factor_Group)] = str(getattr(row, "Aggregation_Method", "equal_weight"))

    l3_dict: dict[str, Any] = {}
    if not layer3_settings.empty and {"Setting", "Value"}.issubset(layer3_settings.columns):
        for row in layer3_settings.dropna(subset=["Setting"]).itertuples(index=False):
            l3_dict[str(row.Setting)] = row.Value

    return {
        "factor_master": fm,
        "group_settings": gs,
        "feature_control": fc,
        "derived_rules": dr,
        "factor_overrides": fo,
        "group_overrides": go,
        "country_region_map": _enabled_frame(country_region),
        "sector_group_map": _enabled_frame(sector_group),
        "sector_factor_interaction": _enabled_frame(interaction),
        "layer3_settings": l3_dict,
        "metas": metas,
        "group_methods": group_methods,
    }


def apply_layer3_excel_settings(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """ExcelのLayer3_Settingsに値がある場合のみConfigを上書きする。"""
    if not settings:
        return config
    mapping = {
        "Estimation_Scope": "primary_scope",
        "Training_Mode": "training_mode",
        "Interaction_Mode": "interaction_mode",
        "Include_Nonlinear_Basis": "include_nonlinear_basis",
        "Include_Sector_Dummy": "include_sector_group_dummy",
        "Include_Sector_Factor_Interaction": "include_sector_factor_interactions",
    }
    for excel_key, cfg_key in mapping.items():
        if excel_key not in settings or pd.isna(settings[excel_key]):
            continue
        value = settings[excel_key]
        if cfg_key.startswith("include_"):
            value = bool(int(value))
        config["layer3"][cfg_key] = value
    return config


def validate_data_columns(data: pd.DataFrame, columns: dict[str, str], metas: dict[str, FactorMeta]) -> None:
    required = set(columns.values()) | set(metas)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"入力データに必須列がありません: {missing}")
    key = [columns["date"], columns["isin"]]
    if data.duplicated(key).any():
        raise ValueError("date + ISINが重複しています。")
