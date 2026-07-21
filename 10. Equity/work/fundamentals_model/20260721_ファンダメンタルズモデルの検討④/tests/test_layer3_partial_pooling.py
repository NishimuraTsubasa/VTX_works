import pandas as pd

from stock_scoring_model.layer3_design_matrix import build_layer3_design


def test_country_deviation_has_stronger_penalty():
    cfg = {"layer3": {
        "include_nonlinear_basis": False,
        "nonlinear_basis": ["linear"],
        "piecewise_knot": 0.0,
        "include_sector_group_dummy": False,
        "include_sector_factor_interactions": False,
        "interaction_mode": "selected_interactions",
        "include_country_controls_in_regional": True,
        "country_deviation_penalty_multiplier": 12.0,
        "country_intercept_penalty_multiplier": 3.0,
    }}
    scores = pd.DataFrame({"Value": [0.1, 0.2, 0.3, 0.4]})
    country = pd.Series(["US", "US", "JP", "JP"])
    region = pd.Series(["Global"] * 4)
    sector = pd.Series(["A"] * 4)
    design = build_layer3_design(scores, country, region, sector, cfg, set(), "hierarchical_partial_pooling")
    penalty = dict(zip(design.X.columns, design.penalty_multipliers))
    assert any(name.startswith("DEV__") and value == 12.0 for name, value in penalty.items())
