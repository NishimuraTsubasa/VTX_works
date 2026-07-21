import pandas as pd

from stock_scoring_model.layer3_design_matrix import build_layer3_design


def base_config():
    return {"layer3": {
        "include_nonlinear_basis": True,
        "nonlinear_basis": ["linear", "piecewise", "quadratic"],
        "piecewise_knot": 0.0,
        "include_sector_group_dummy": True,
        "include_sector_factor_interactions": True,
        "interaction_mode": "selected_interactions",
        "include_country_controls_in_regional": True,
        "country_deviation_penalty_multiplier": 10.0,
        "country_intercept_penalty_multiplier": 2.0,
    }}


def test_selected_interaction_is_created():
    scores = pd.DataFrame({"Value": [0.2, 0.4], "Momentum": [0.1, 0.3]})
    country = pd.Series(["US", "JP"])
    region = pd.Series(["NA", "ASIA"])
    sector = pd.Series(["Banks", "Growth"])
    design = build_layer3_design(scores, country, region, sector, base_config(), {("Banks", "Value")}, "country_independent")
    assert any(c.startswith("INT__Banks__Value") for c in design.X.columns)
    assert not any(c.startswith("INT__Banks__Momentum") for c in design.X.columns)
