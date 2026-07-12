from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman"))


def safe_pearson(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson"))


def rank_to_unit_interval(s: pd.Series) -> pd.Series:
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    n = int(valid.sum())
    if n == 0:
        return out
    out.loc[valid] = (s.loc[valid].rank(method="average") - 0.5) / n
    return out


def flatten_dict(data: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(flatten_dict(value, full))
        elif isinstance(value, list):
            rows.append((full, json.dumps(value, ensure_ascii=False)))
        else:
            rows.append((full, value))
    return rows


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
