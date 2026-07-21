from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_scoring_model.binscatter_runner import run_binscatter_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-averaged binscatter診断を生成")
    parser.add_argument("--config", default=str(ROOT / "config" / "model_config.py"))
    args = parser.parse_args()
    result = run_binscatter_pipeline(args.config)
    print(f"Binscatter completed: {result['output_dir']}", flush=True)
    # 一部環境でMatplotlib/BLASの終了処理が残るため、PDF・Excel保存完了後に即時終了する。
    os._exit(0)


if __name__ == "__main__":
    main()
