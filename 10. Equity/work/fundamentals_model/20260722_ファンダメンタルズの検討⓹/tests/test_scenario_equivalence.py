from __future__ import annotations

import numpy as np
import pandas as pd

from stock_scoring_model.master import FactorMeta
from stock_scoring_model.scenarios import _hierarchical_factor_count_prediction


def test_factor_count_hierarchy_reproduces_direct_equal_weight() -> None:
    scores = pd.DataFrame({"FA1": [0.1, np.nan, 0.3], "FA2": [0.5, 0.2, 0.1], "FA3": [-0.2, 0.4, 0.5]})
    metas = {
        "FA1": FactorMeta("FA1", "Value", 1, 1.0),
        "FA2": FactorMeta("FA2", "Value", 1, 1.0),
        "FA3": FactorMeta("FA3", "Momentum", 1, 1.0),
    }
    expected = scores.mean(axis=1, skipna=True)
    actual = _hierarchical_factor_count_prediction(scores, metas)
    assert np.allclose(expected, actual, equal_nan=True)
