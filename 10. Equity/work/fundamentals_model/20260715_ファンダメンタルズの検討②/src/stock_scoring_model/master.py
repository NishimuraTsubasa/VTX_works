from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
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
    out = df.copy()
    out = out.dropna(how="all")
    out.columns = [str(c).strip() for c in out.columns]
    return out


def parse_master(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    fm = _clean_frame(sheets.get("Factor_Master"))
    gs = _clean_frame(sheets.get("Group_Settings"))
    fc = _clean_frame(sheets.get("Feature_Engineering_Control"))
    dr = _clean_frame(sheets.get("Derived_Feature_Rules"))
    fo = _clean_frame(sheets.get("Factor_Overrides"))
    go = _clean_frame(sheets.get("Group_Overrides"))

    required = {"Factor_Code", "Factor_Group", "Enabled", "Direction", "Base_Weight"}
    missing = required - set(fm.columns)
    if missing:
        raise ValueError(f"Factor_Masterの必須列が不足しています: {sorted(missing)}")

    fm = fm[fm["Factor_Code"].notna()].copy()
    fm["Factor_Code"] = fm["Factor_Code"].astype(str).str.strip()
    if fm["Factor_Code"].duplicated().any():
        dup = fm.loc[fm["Factor_Code"].duplicated(), "Factor_Code"].tolist()
        raise ValueError(f"Factor_Codeが重複しています: {dup}")
    fm["Enabled"] = pd.to_numeric(fm["Enabled"], errors="coerce").fillna(0).astype(int)
    fm["Direction"] = pd.to_numeric(fm["Direction"], errors="coerce").fillna(1).astype(int)
    if not fm["Direction"].isin([-1, 1]).all():
        raise ValueError("Directionは1または-1のみです。")
    fm["Base_Weight"] = pd.to_numeric(fm["Base_Weight"], errors="coerce").fillna(1.0)

    if not gs.empty:
        gs = gs[gs["Factor_Group"].notna()].copy()
        gs["Factor_Group"] = gs["Factor_Group"].astype(str).str.strip()
        gs["Enabled"] = pd.to_numeric(gs["Enabled"], errors="coerce").fillna(0).astype(int)
    enabled_groups = set(gs.loc[gs.get("Enabled", 1).eq(1), "Factor_Group"]) if not gs.empty else set(fm["Factor_Group"])
    enabled = fm[(fm["Enabled"] == 1) & fm["Factor_Group"].isin(enabled_groups)].copy()

    metas = {
        row.Factor_Code: FactorMeta(
            code=row.Factor_Code,
            group=str(row.Factor_Group),
            direction=int(row.Direction),
            base_weight=float(row.Base_Weight),
        )
        for row in enabled.itertuples(index=False)
    }
    group_methods = {}
    if not gs.empty:
        for row in gs.itertuples(index=False):
            if int(getattr(row, "Enabled", 1)) == 1:
                group_methods[str(row.Factor_Group)] = str(getattr(row, "Aggregation_Method", "equal_weight"))

    return {
        "factor_master": fm,
        "group_settings": gs,
        "feature_control": fc,
        "derived_rules": dr,
        "factor_overrides": fo,
        "group_overrides": go,
        "metas": metas,
        "group_methods": group_methods,
    }


def validate_data_columns(data: pd.DataFrame, columns: dict[str, str], metas: dict[str, FactorMeta]) -> None:
    required = set(columns.values()) | set(metas)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"入力データに必須列がありません: {missing}")
    key = [columns["date"], columns["isin"]]
    if data.duplicated(key).any():
        raise ValueError("date + ISINが重複しています。")
