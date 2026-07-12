from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    settings: dict[str, Any]
    run_control: dict[str, Any]
    futures: pd.DataFrame
    futures_fields: pd.DataFrame
    cross_asset: pd.DataFrame
    overrides: pd.DataFrame
    output_mapping: pd.DataFrame


def _read_key_value(path: Path, sheet: str) -> dict[str, Any]:
    frame = pd.read_excel(path, sheet_name=sheet)
    if frame.empty:
        return {}
    return dict(zip(frame.iloc[:, 0].astype(str), frame.iloc[:, 1]))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_config(path: Path) -> PreprocessConfig:
    settings = _read_key_value(path, "Settings")
    run_control = _read_key_value(path, "Run_Control")
    futures = pd.read_excel(path, sheet_name="Futures_Universe")
    futures_fields = pd.read_excel(path, sheet_name="Futures_Fields")
    cross_asset = pd.read_excel(path, sheet_name="Cross_Asset")
    overrides = pd.read_excel(path, sheet_name="Overrides")
    output_mapping = pd.read_excel(path, sheet_name="Output_Mapping")

    for frame in (futures, futures_fields, cross_asset, overrides):
        if "active" in frame.columns:
            frame["active"] = frame["active"].map(_as_bool)
    if "required" in futures.columns:
        futures["required"] = futures["required"].map(_as_bool)
    if "required" in futures_fields.columns:
        futures_fields["required"] = futures_fields["required"].map(_as_bool)
    if "required" in cross_asset.columns:
        cross_asset["required"] = cross_asset["required"].map(_as_bool)

    return PreprocessConfig(
        settings=settings,
        run_control=run_control,
        futures=futures,
        futures_fields=futures_fields,
        cross_asset=cross_asset,
        overrides=overrides,
        output_mapping=output_mapping,
    )


def parse_config_date(value: Any, fallback: date | None = None) -> date:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        if fallback is None:
            raise ValueError("Date is missing and no fallback was supplied")
        return fallback
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if text.upper() in {"TODAY", "AUTO"}:
        if fallback is None:
            return date.today()
        return fallback
    return pd.Timestamp(text).date()
