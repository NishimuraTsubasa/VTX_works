from __future__ import annotations

import pandas as pd

from .nonlinear_basis import factor_group_from_basis


def add_sector_dummies(base: pd.DataFrame, sector_group: pd.Series, drop_first: bool = True) -> pd.DataFrame:
    dummies = pd.get_dummies(sector_group.astype(str), prefix="SEC", drop_first=drop_first, dtype=float)
    return pd.concat([base, dummies], axis=1)


def add_country_dummies(base: pd.DataFrame, country: pd.Series, drop_first: bool = True) -> pd.DataFrame:
    dummies = pd.get_dummies(country.astype(str), prefix="CTRY", drop_first=drop_first, dtype=float)
    return pd.concat([base, dummies], axis=1)


def add_sector_factor_interactions(
    base_basis: pd.DataFrame,
    sector_group: pd.Series,
    mode: str,
    selected: set[tuple[str, str]],
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    sectors = sorted(sector_group.dropna().astype(str).unique())
    for sec in sectors:
        mask = sector_group.astype(str).eq(sec).astype(float)
        for col in base_basis.columns:
            group = factor_group_from_basis(col)
            if mode == "selected_interactions" and (sec, group) not in selected:
                continue
            columns[f"INT__{sec}__{col}"] = base_basis[col] * mask
    return pd.DataFrame(columns, index=base_basis.index)


def add_country_deviation_features(base_features: pd.DataFrame, country: pd.Series) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    countries = sorted(country.dropna().astype(str).unique())
    candidate_cols = [c for c in base_features.columns if not c.startswith("SEC_") and not c.startswith("CTRY_")]
    for ctry in countries:
        mask = country.astype(str).eq(ctry).astype(float)
        for col in candidate_cols:
            columns[f"DEV__{ctry}__{col}"] = base_features[col] * mask
    return pd.DataFrame(columns, index=base_features.index)
