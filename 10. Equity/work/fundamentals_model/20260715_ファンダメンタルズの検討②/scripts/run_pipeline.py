from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_scoring_model.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "model_config.py"))
    args = parser.parse_args()
    result = run_pipeline(args.config)
    print(f"Completed: {result['output_dirs']['root']}")


if __name__ == "__main__":
    main()
