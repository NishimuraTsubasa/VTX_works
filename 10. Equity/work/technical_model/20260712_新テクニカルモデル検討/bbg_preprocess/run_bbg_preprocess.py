from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from bbg_preprocess.pipeline import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental Bloomberg download and model-input preprocessing")
    parser.add_argument("--config", type=Path, default=HERE / "config" / "BBG_Config.xlsx")
    args = parser.parse_args()
    outputs = run(args.config)
    print("Bloomberg preprocessing completed.")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
