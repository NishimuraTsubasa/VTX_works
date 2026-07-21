from __future__ import annotations

import argparse

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="個別銘柄スコアリングモデル分析")
    parser.add_argument("--config", default="config/model_config.py")
    args = parser.parse_args()
    result = run_pipeline(args.config)
    print(f"Completed. scenarios={result['scenario_count']} output={result['output_dirs']['root']}")


if __name__ == "__main__":
    main()
