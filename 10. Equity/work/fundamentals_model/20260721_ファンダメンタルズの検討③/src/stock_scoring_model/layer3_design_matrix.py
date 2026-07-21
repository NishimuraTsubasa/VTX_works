from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .interaction_features import (
    add_country_deviation_features,
    add_country_dummies,
    add_sector_dummies,
    add_sector_factor_interactions,
)
from .nonlinear_basis import basis_frame


@dataclass
class Layer3Design:
    X: pd.DataFrame
    penalty_multipliers: np.ndarray
    feature_types: pd.DataFrame
    eligible_rows: pd.Series


def build_layer3_design(
    factor_scores: pd.DataFrame,
    country: pd.Series,
    region: pd.Series,
    sector_group: pd.Series,
    config: dict[str, Any],
    selected_interaction_pairs: set[tuple[str, str]],
    scope: str,
) -> Layer3Design:
    cfg = config["layer3"]
    basis_types = cfg.get("nonlinear_basis", ["linear", "piecewise", "quadratic"])
    if not cfg.get("include_nonlinear_basis", True):
        basis_types = ["linear"]
    minimum_coverage = float(cfg.get("minimum_factor_score_coverage", 0.50))
    eligible_rows = factor_scores.notna().mean(axis=1).ge(minimum_coverage)
    basis = basis_frame(factor_scores, basis_types, knot=float(cfg.get("piecewise_knot", 0.0)))
    frames = [basis]
    type_rows = [{"Feature": c, "FeatureType": "factor_basis"} for c in basis.columns]

    if cfg.get("include_sector_group_dummy", True):
        sector_part = add_sector_dummies(pd.DataFrame(index=basis.index), sector_group, drop_first=True)
        frames.append(sector_part)
        type_rows += [{"Feature": c, "FeatureType": "sector_dummy"} for c in sector_part.columns]

    interaction_part = pd.DataFrame(index=basis.index)
    if cfg.get("include_sector_factor_interactions", True):
        interaction_part = add_sector_factor_interactions(
            basis,
            sector_group,
            str(cfg.get("interaction_mode", "selected_interactions")),
            selected_interaction_pairs,
        )
        frames.append(interaction_part)
        type_rows += [{"Feature": c, "FeatureType": "sector_factor_interaction"} for c in interaction_part.columns]

    base = pd.concat(frames, axis=1).astype(float)
    penalties = pd.Series(1.0, index=base.columns)

    if scope == "regional_pooling" and cfg.get("include_country_controls_in_regional", True):
        country_part = add_country_dummies(pd.DataFrame(index=base.index), country, drop_first=True)
        base = pd.concat([base, country_part], axis=1)
        penalties = pd.concat([penalties, pd.Series(1.0, index=country_part.columns)])
        type_rows += [{"Feature": c, "FeatureType": "country_control"} for c in country_part.columns]

    if scope == "hierarchical_partial_pooling":
        # 地域共通係数 + 国固有補正。国固有補正は強い罰則を付ける。
        deviation = add_country_deviation_features(pd.concat([basis, interaction_part], axis=1), country)
        country_intercept = pd.get_dummies(country.astype(str), prefix="CTRY_INT", drop_first=True, dtype=float)
        base = pd.concat([base, country_intercept, deviation], axis=1)
        penalties = pd.concat([
            penalties,
            pd.Series(float(cfg.get("country_intercept_penalty_multiplier", 2.0)), index=country_intercept.columns),
            pd.Series(float(cfg.get("country_deviation_penalty_multiplier", 10.0)), index=deviation.columns),
        ])
        type_rows += [{"Feature": c, "FeatureType": "country_intercept"} for c in country_intercept.columns]
        type_rows += [{"Feature": c, "FeatureType": "country_deviation"} for c in deviation.columns]

    # 欠損は列中央値。全欠損列は0。
    base = base.replace([np.inf, -np.inf], np.nan)
    base = base.fillna(base.median()).fillna(0.0)
    penalties = penalties.reindex(base.columns).fillna(1.0)
    return Layer3Design(
        X=base,
        penalty_multipliers=penalties.to_numpy(float),
        feature_types=pd.DataFrame(type_rows).drop_duplicates("Feature"),
        eligible_rows=eligible_rows.reindex(base.index).fillna(False),
    )
