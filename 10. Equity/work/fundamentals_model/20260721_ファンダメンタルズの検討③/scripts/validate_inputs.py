from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_scoring_model.config_loader import load_config
from stock_scoring_model.io import read_inputs
from stock_scoring_model.master import parse_master, validate_data_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "model_config.py"))
    args = parser.parse_args()
    config, root = load_config(args.config)
    data, sheets, _ = read_inputs(config, root)
    parsed = parse_master(sheets)
    validate_data_columns(data, config["columns"], parsed["metas"])

    c = config["columns"]
    countries = set(data[c["country"]].dropna().astype(str))
    mapped_countries = set(parsed["country_region_map"].get("Country", []).astype(str)) if not parsed["country_region_map"].empty else set()
    sectors = set(data[c["sector"]].dropna().astype(str))
    mapped_sectors = set(parsed["sector_group_map"].get("Sector", []).astype(str)) if not parsed["sector_group_map"].empty else set()

    print("Input validation passed.")
    print(f"Rows: {len(data):,}")
    print(f"Dates: {data[c['date']].nunique():,}")
    print(f"ISINs: {data[c['isin']].nunique():,}")
    print(f"Enabled factors: {len(parsed['metas']):,}")
    if countries - mapped_countries:
        print(f"WARNING unmapped countries: {sorted(countries - mapped_countries)}")
    if sectors - mapped_sectors:
        print(f"WARNING unmapped sectors: {sorted(sectors - mapped_sectors)}")


if __name__ == "__main__":
    main()
