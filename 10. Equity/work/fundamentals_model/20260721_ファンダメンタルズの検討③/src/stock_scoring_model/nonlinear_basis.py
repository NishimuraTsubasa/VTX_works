from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def basis_frame(
    values: pd.DataFrame,
    basis_types: Iterable[str],
    knot: float = 0.0,
) -> pd.DataFrame:
    """FactorScore列から線形・区分線形・二次基底を生成する。"""
    result = pd.DataFrame(index=values.index)
    for col in values.columns:
        x = pd.to_numeric(values[col], errors="coerce")
        if "linear" in basis_types:
            result[f"{col}__LIN"] = x
        if "piecewise" in basis_types:
            result[f"{col}__HINGE"] = np.maximum(x - knot, 0.0)
        if "quadratic" in basis_types:
            # 線形項との相関を少し抑えるため月内中心化は呼出側で必要に応じて行う。
            result[f"{col}__QUAD"] = x**2
    return result


def factor_group_from_basis(column: str) -> str:
    return str(column).split("__", 1)[0]
