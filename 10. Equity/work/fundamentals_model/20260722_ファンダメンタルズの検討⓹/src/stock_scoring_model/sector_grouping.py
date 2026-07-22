from __future__ import annotations

from typing import Any

import pandas as pd


def apply_country_region_map(
    data: pd.DataFrame,
    country_col: str,
    mapping: pd.DataFrame,
) -> pd.Series:
    if mapping.empty:
        return data[country_col].astype(str)
    mp = mapping.copy()
    mp = mp[mp.get("Enabled", 1).fillna(1).astype(int).eq(1)]
    lookup = dict(zip(mp["Country"].astype(str), mp["Region"].astype(str)))
    return data[country_col].astype(str).map(lookup).fillna("Unmapped")


def apply_sector_group_map(
    data: pd.DataFrame,
    sector_col: str,
    mapping: pd.DataFrame,
) -> pd.Series:
    if mapping.empty:
        return data[sector_col].astype(str)
    mp = mapping.copy()
    mp = mp[mp.get("Enabled", 1).fillna(1).astype(int).eq(1)]
    lookup = dict(zip(mp["Sector"].astype(str), mp["Sector_Group"].astype(str)))
    return data[sector_col].astype(str).map(lookup).fillna("Other")


def selected_interactions(mapping: pd.DataFrame) -> set[tuple[str, str]]:
    if mapping.empty:
        return set()
    mp = mapping.copy()
    mp = mp[mp.get("Enabled", 1).fillna(1).astype(int).eq(1)]
    return set(zip(mp["Sector_Group"].astype(str), mp["Factor_Group"].astype(str)))
