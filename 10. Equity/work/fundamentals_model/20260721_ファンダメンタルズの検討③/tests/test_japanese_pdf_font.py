from __future__ import annotations

from matplotlib import pyplot as plt

from stock_scoring_model.reporting import _setup_matplotlib


def test_japanese_font_setup_embeds_truetype():
    selected = _setup_matplotlib()
    assert isinstance(selected, str) and selected
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["axes.unicode_minus"] is False
