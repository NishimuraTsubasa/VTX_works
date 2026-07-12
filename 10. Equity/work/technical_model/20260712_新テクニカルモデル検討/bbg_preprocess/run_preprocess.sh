#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python run_bbg_preprocess.py --config config/BBG_Config.xlsx
