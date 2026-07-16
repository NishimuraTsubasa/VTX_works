from __future__ import annotations

import pickle
import sys
from pathlib import Path

from .diagnostics import (
    create_factor_bin_pdf,
    create_factor_performance_pdf,
    create_factor_model_selection_pdf,
    create_factor_scatter_pdf,
    create_scenario_quintile_cumulative_pdf,
    create_scenario_comparison_pdf,
)

FUNCTIONS = {
    "create_factor_scatter_pdf": create_factor_scatter_pdf,
    "create_factor_bin_pdf": create_factor_bin_pdf,
    "create_factor_performance_pdf": create_factor_performance_pdf,
    "create_factor_model_selection_pdf": create_factor_model_selection_pdf,
    "create_scenario_quintile_cumulative_pdf": create_scenario_quintile_cumulative_pdf,
    "create_scenario_comparison_pdf": create_scenario_comparison_pdf,
}


def main(payload_path: str) -> None:
    with open(payload_path, "rb") as f:
        payload = pickle.load(f)
    func = FUNCTIONS[payload["function"]]
    func(*payload["args"], Path(payload["output_path"]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m stock_scoring_model.report_worker PAYLOAD.pkl")
    main(sys.argv[1])
