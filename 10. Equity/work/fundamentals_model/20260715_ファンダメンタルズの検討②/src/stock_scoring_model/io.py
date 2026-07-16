from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .factor_master import FactorSettingsBundle, load_factor_settings, active_factor_codes

LOGGER = logging.getLogger(__name__)


@dataclass
class DataBundle:
    stocks: pd.DataFrame
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


def load_all(config: dict, config_path: str | Path) -> DataBundle:
    base_dir = Path(config_path).resolve().parent.parent
    stocks = load_stock_data(config, base_dir)
    factor_master_path = _resolve(base_dir, config["data"]["factor_master_file"])
    factor_settings = load_factor_settings(
        factor_master_path,
        config["data"].get("factor_master_sheet_map", {}),
        stock_columns=list(stocks.columns),
    )
    return DataBundle(stocks=stocks, factor_settings=factor_settings)


def infer_factor_columns(stocks: pd.DataFrame, factor_settings: FactorSettingsBundle, config: dict) -> list[str]:
    factors = active_factor_codes(factor_settings.factor_master, factor_settings.group_settings)
    missing = [c for c in factors if c not in stocks.columns]
    if missing and config["factors"].get("require_all_configured_factors", True):
        raise ValueError(f"Enabled factor columns not found: {missing}")
    factors = [c for c in factors if c in stocks.columns]
    configured = set(factor_settings.factor_master["Factor_Code"].astype(str))
    unknown = [c for c in stocks.columns if str(c).startswith("FA") and c not in configured]
    if unknown and config["factors"].get("reject_unknown_factor_columns", False):
        raise ValueError(f"Input factor columns are not configured in factor_master.xlsx: {unknown}")
    return factors
