from __future__ import annotations

import numpy as np

from stock_scoring_model.binscatter import fit_regressions


def test_quadratic_fit_has_high_r2_for_quadratic_data() -> None:
    x = np.linspace(-2, 2, 20)
    y = 0.01 + 0.02 * x + 0.03 * x**2
    cfg = {
        "binscatter": {
            "regressions": {
                "linear": True,
                "quadratic": True,
                "broken_stick": True,
                "broken_stick_knot": "median",
            }
        }
    }
    fits = fit_regressions(x, y, cfg)
    assert fits["quadratic"].r2 > 0.999
    assert fits["quadratic"].r2 >= fits["linear"].r2
