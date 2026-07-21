import numpy as np

from stock_scoring_model.layer1_single_factor import design_matrix, fit_single_factor


def test_candidate_design_shapes():
    x = np.array([-1.0, 0.0, 1.0])
    assert design_matrix(x, "linear").shape == (3, 1)
    assert design_matrix(x, "piecewise").shape == (3, 2)
    assert design_matrix(x, "quadratic").shape == (3, 2)


def test_linear_model_prediction():
    x = np.arange(10, dtype=float)
    y = 2.0 * x + 1.0
    model = fit_single_factor(x, y, "linear", ridge_alpha=0.0)
    assert np.allclose(model.predict(x), y)
