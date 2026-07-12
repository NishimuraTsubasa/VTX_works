from __future__ import annotations

import argparse

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stock-to-index scoring model.")
    parser.add_argument(
        "--config",
        default="config/model_config.py",
        help="Path to Python configuration file defining CONFIG = {...}.",
    )
    args = parser.parse_args()
    outputs = run_pipeline(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
