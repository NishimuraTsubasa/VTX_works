from __future__ import annotations

import argparse

from .binscatter_runner import run_binscatter_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-averaged binscatter diagnostics")
    parser.add_argument("--config", default="config/model_config.py")
    args = parser.parse_args()
    result = run_binscatter_pipeline(args.config)
    print(f"Completed. output={result['output_dir']}")


if __name__ == "__main__":
    main()
