from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .factor_master import FactorSettingsBundle, load_factor_settings, active_factor_codes

LOGGER = logging.getLogger(__name__)


@dataclass
class DataBundle:
    stocks: pd.DataFrame
    constituents: pd.DataFrame
    sector_weights: pd.DataFrame
    futures_returns: pd.DataFrame
    factor_settings: FactorSettingsBundle


def _resolve(base_dir: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_stock_data(config: dict, base_dir: Path) -> pd.DataFrame:
    path = _resolve(base_dir, config["data"]["factors_file"])
    sheet = config["data"]["factors_sheet"]
    df = pd.read_excel(path, sheet_name=sheet)
    date_col = config["columns"]["date"]
    isin_col = config["columns"]["isin"]
    if date_col not in df or isin_col not in df:
        raise ValueError(f"Stock file requires columns: {date_col}, {isin_col}")
    df[date_col] = pd.to_datetime(df[date_col])
    df[isin_col] = df[isin_col].astype(str).str.strip()
    return df


def load_constituents(config: dict, base_dir: Path) -> pd.DataFrame:
    path = _resolve(base_dir, config["data"]["constituents_file"])
    xls = pd.ExcelFile(path)
    ignore = set(config["data"].get("ignore_constituent_sheets", []))
    frames: list[pd.DataFrame] = []
    isin_col = config["columns"]["isin"]
    sector_col = config["columns"]["sector"]
    country_col = config["columns"]["country"]
    effective_col = config["columns"]["effective_date"]
    country_map = config["data"].get("index_country_map", {})

    for sheet in xls.sheet_names:
        if sheet in ignore or sheet.startswith("_"):
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        if df.empty:
            continue
        if isin_col not in df or sector_col not in df:
            raise ValueError(
                f"Constituent sheet '{sheet}' requires columns: {isin_col}, {sector_col}"
            )
        df = df.copy()
        df[isin_col] = df[isin_col].astype(str).str.strip()
        df[sector_col] = df[sector_col].astype(str).str.strip()
        df["index_name"] = sheet
        if country_col not in df:
            df[country_col] = country_map.get(sheet, sheet)
        if effective_col in df:
            df[effective_col] = pd.to_datetime(df[effective_col])
        frames.append(df)

    if not frames:
        raise ValueError("No index constituent sheets were found.")
    return pd.concat(frames, ignore_index=True)


def load_sector_weights(config: dict, base_dir: Path) -> pd.DataFrame:
    path = _resolve(base_dir, config["data"]["sector_weights_file"])
    sheet = config["data"]["sector_weights_sheet"]
    wide = pd.read_excel(path, sheet_name=sheet)
    sector_col = config["columns"]["sector_weights_sector"]
    if sector_col not in wide:
        raise ValueError(f"Sector weights file requires '{sector_col}' as the first key column.")
    long = wide.melt(id_vars=[sector_col], var_name="index_name", value_name="sector_weight")
    long[sector_col] = long[sector_col].astype(str).str.strip()
    long["sector_weight"] = pd.to_numeric(long["sector_weight"], errors="coerce")
    long = long.dropna(subset=["sector_weight"])

    totals = long.groupby("index_name")["sector_weight"].transform("sum")
    percent_mask = (totals > 1.5) & (totals <= 101.0)
    long.loc[percent_mask, "sector_weight"] = long.loc[percent_mask, "sector_weight"] / 100.0
    if config["aggregation"].get("normalize_sector_weights", True):
        totals = long.groupby("index_name")["sector_weight"].transform("sum")
        long["sector_weight"] = long["sector_weight"] / totals.where(totals > 0, np.nan)
    return long.reset_index(drop=True)


def load_futures_returns(config: dict, base_dir: Path) -> pd.DataFrame:
    path = _resolve(base_dir, config["data"]["futures_file"])
    frequency = config["data"]["frequency"]
    sheet = config["data"]["futures_sheet_map"][frequency]
    df = pd.read_excel(path, sheet_name=sheet)
    date_col = config["columns"]["date"]
    if date_col not in df:
        raise ValueError(f"Futures return sheet requires '{date_col}'.")
    df[date_col] = pd.to_datetime(df[date_col])
    for col in df.columns:
        if col != date_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(date_col).reset_index(drop=True)


def load_all(config: dict, config_path: str | Path) -> DataBundle:
    base_dir = Path(config_path).resolve().parent.parent
    stocks = load_stock_data(config, base_dir)
    factor_master_path = _resolve(base_dir, config["data"]["factor_master_file"])
    factor_settings = load_factor_settings(
        factor_master_path,
        config["data"].get("factor_master_sheet_map", {}),
        stock_columns=list(stocks.columns),
    )
    return DataBundle(
        stocks=stocks,
        constituents=load_constituents(config, base_dir),
        sector_weights=load_sector_weights(config, base_dir),
        futures_returns=load_futures_returns(config, base_dir),
        factor_settings=factor_settings,
    )


def infer_factor_columns(stocks: pd.DataFrame, factor_settings: FactorSettingsBundle, config: dict) -> list[str]:
    factors = active_factor_codes(factor_settings.factor_master, factor_settings.group_settings)
    missing = [c for c in factors if c not in stocks.columns]
    if missing and config["factors"].get("require_all_configured_factors", True):
        raise ValueError(f"Enabled factor columns not found: {missing}")
    factors = [c for c in factors if c in stocks.columns]
    unknown = [c for c in stocks.columns if str(c).startswith("FA") and c not in set(factor_settings.factor_master["Factor_Code"])]
    if unknown and config["factors"].get("reject_unknown_factor_columns", False):
        raise ValueError(f"Input factor columns are not configured in factor_master.xlsx: {unknown}")
    return factors


def attach_security_attributes(stocks: pd.DataFrame, constituents: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = config["columns"]
    isin_col, sector_col, country_col = cols["isin"], cols["sector"], cols["country"]
    issues: list[dict] = []

    attr = constituents[[isin_col, sector_col, country_col]].drop_duplicates()
    conflicts = attr.groupby(isin_col).agg(
        sector_count=(sector_col, "nunique"), country_count=(country_col, "nunique")
    )
    for isin, row in conflicts[(conflicts.sector_count > 1) | (conflicts.country_count > 1)].iterrows():
        issues.append({
            "issue_type": "attribute_conflict",
            "ISIN": isin,
            "detail": f"sector_count={row.sector_count}, country_count={row.country_count}",
        })

    # Resolve conflicts by the most frequent value; retain an issue log.
    sector_map = constituents.groupby(isin_col)[sector_col].agg(lambda x: x.mode().iloc[0])
    country_map = constituents.groupby(isin_col)[country_col].agg(lambda x: x.mode().iloc[0])
    mapping = pd.DataFrame({isin_col: sector_map.index, "_sector_map": sector_map.values, "_country_map": country_map.values})

    out = stocks.merge(mapping, on=isin_col, how="left")
    if sector_col not in out:
        out[sector_col] = out["_sector_map"]
    else:
        out[sector_col] = out[sector_col].fillna(out["_sector_map"])
    if country_col not in out:
        out[country_col] = out["_country_map"]
    else:
        out[country_col] = out[country_col].fillna(out["_country_map"])
    out = out.drop(columns=["_sector_map", "_country_map"])
    return out, pd.DataFrame(issues)
