from __future__ import annotations

import pickle
import sys
from pathlib import Path

from .diagnostics import (
    create_factor_bin_pdf,
    create_factor_performance_pdf,
    create_factor_model_selection_pdf,
    create_factor_scatter_pdf,
    create_futures_risk_pdf,
    create_index_exposure_pdf,
    create_index_factor_trends_pdf,
    create_model_accuracy_pdf,
    create_universe_selection_pdf,
)

FUNCTIONS = {
    "create_factor_scatter_pdf": create_factor_scatter_pdf,
    "create_factor_bin_pdf": create_factor_bin_pdf,
    "create_factor_performance_pdf": create_factor_performance_pdf,
    "create_factor_model_selection_pdf": create_factor_model_selection_pdf,
    "create_index_exposure_pdf": create_index_exposure_pdf,
    "create_index_factor_trends_pdf": create_index_factor_trends_pdf,
    "create_model_accuracy_pdf": create_model_accuracy_pdf,
    "create_universe_selection_pdf": create_universe_selection_pdf,
    "create_futures_risk_pdf": create_futures_risk_pdf,
}


def main(payload_path: str) -> None:
    with open(payload_path, "rb") as f:
        payload = pickle.load(f)
    name = payload["function"]
    func = FUNCTIONS[name]
    args = payload["args"]
    output_path = Path(payload["output_path"])
    func(*args, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m stock_index_model.report_worker PAYLOAD.pkl")
    main(sys.argv[1])
