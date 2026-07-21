import pandas as pd

from stock_scoring_model.layer2_factor_aggregation import aggregate_layer2_factor_scores
from stock_scoring_model.master import FactorMeta


def test_equal_weight_group_aggregation():
    data = pd.DataFrame({"date": [1, 1], "NextMonthReturn": [0.1, -0.1]})
    subs = pd.DataFrame({"FA1": [1.0, 3.0], "FA2": [3.0, 1.0]})
    metas = {"FA1": FactorMeta("FA1", "Value", 1, 1.0), "FA2": FactorMeta("FA2", "Value", 1, 1.0)}
    cfg = {"columns": {"date": "date"}, "layer2": {}}
    result, _ = aggregate_layer2_factor_scores(data, subs, metas, {"Value": "equal_weight"}, cfg)
    assert result["Value"].tolist() == [2.0, 2.0]
