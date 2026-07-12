#!/usr/bin/env bash
set -euo pipefail
python run_pipeline.py "$@"
python src/build_excel_summary.py
