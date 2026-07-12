from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stock_index_model.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="株式ファクター・指数先物分析パイプライン")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "model_config.py",
        help="Python辞書形式のConfigファイル",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = run_pipeline(args.config)
    for key, path in outputs.items():
        print(f"{key}: {path}")
