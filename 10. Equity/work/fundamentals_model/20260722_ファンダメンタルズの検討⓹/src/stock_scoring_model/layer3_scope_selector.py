from __future__ import annotations

from typing import Any

import pandas as pd

from .layer3_country_model import fit_country_models
from .layer3_design_matrix import build_layer3_design
from .layer3_partial_pooling import fit_partial_pooling_models
from .layer3_regional_model import fit_regional_models
from .sector_grouping import selected_interactions


def run_layer3_scopes(
    data: pd.DataFrame,
    factor_scores: pd.DataFrame,
    region: pd.Series,
    sector_group: pd.Series,
    interaction_map: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, dict[str, pd.DataFrame | pd.Series]]:
    country = data[config["columns"]["country"]].astype(str)
    selected = selected_interactions(interaction_map)
    outputs: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
    scopes = list(dict.fromkeys(config["layer3"].get("comparison_scopes", [config["layer3"].get("primary_scope", "country_independent")])))

    for scope in scopes:
        design = build_layer3_design(
            factor_scores,
            country,
            region,
            sector_group,
            config,
            selected,
            scope,
        )
        if scope == "country_independent":
            result = fit_country_models(data, design.X, design.penalty_multipliers, design.standardize_mask, config, eligible_rows=design.eligible_rows)
        elif scope == "regional_pooling":
            result = fit_regional_models(data, design.X, design.penalty_multipliers, design.standardize_mask, region, config, eligible_rows=design.eligible_rows)
        elif scope == "hierarchical_partial_pooling":
            result = fit_partial_pooling_models(data, design.X, design.penalty_multipliers, design.standardize_mask, region, config, eligible_rows=design.eligible_rows)
        else:
            raise ValueError(f"Unsupported layer3 scope: {scope}")
        outputs[scope] = {
            "Prediction": result.prediction,
            "CoefficientHistory": result.coefficient_history,
            "ModelHistory": result.model_history,
            "FeatureTypes": design.feature_types,
            "StandardizeMask": pd.Series(design.standardize_mask, index=design.X.columns, name="Standardize"),
            "EligibleRows": design.eligible_rows,
        }
    return outputs
