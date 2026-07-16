from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "input"
TEMPLATES = ROOT / "data" / "templates"
OUT.mkdir(parents=True, exist_ok=True)
TEMPLATES.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
markets = {
    "Japan": {"prefix": "JP", "currency": "JPY"},
    "United States": {"prefix": "US", "currency": "USD"},
    "Europe": {"prefix": "EU", "currency": "EUR"},
}
sectors = ["Financials", "Industrials", "Technology", "Consumer"]

securities: list[dict] = []
for country, meta in markets.items():
    for n in range(24):
        securities.append({
            "country": country,
            "currency": meta["currency"],
            "ISIN": f"{meta['prefix']}{n:010d}",
            "sector": sectors[n % len(sectors)],
            "market_cap_base": float(np.exp(rng.normal(24.5, 1.0))),
        })
sec = pd.DataFrame(securities)
dates = pd.date_range("2017-01-31", periods=36, freq="ME")
latent = {isin: rng.normal(size=4) for isin in sec["ISIN"]}
country_shocks = {country: rng.normal(0.004, 0.025, len(dates)) for country in markets}
sector_shocks = {(country, sector): rng.normal(0.0, 0.015, len(dates)) for country in markets for sector in sectors}

rows: list[dict] = []
for t, date in enumerate(dates):
    regime = np.sin(t / 12.0)
    for r in sec.itertuples(index=False):
        base = latent[r.ISIN]
        value = 0.65 * base[0] + 0.35 * rng.normal()
        quality = 0.65 * base[1] + 0.35 * rng.normal()
        momentum = 0.50 * base[2] + 0.50 * rng.normal() + 0.15 * regime
        growth = 0.60 * base[3] + 0.40 * rng.normal()
        stock_return = (
            country_shocks[r.country][t]
            + sector_shocks[(r.country, r.sector)][t]
            + 0.0030 * value
            + 0.0025 * quality
            + 0.0035 * momentum
            - 0.0015 * (growth**2 - 1)
            + rng.normal(0, 0.035)
        )
        rows.append({
            "date": date,
            "ISIN": r.ISIN,
            "stock_return": stock_return,
            "market_cap": r.market_cap_base * np.exp(rng.normal(0, 0.08)),
            "currency": r.currency,
            "sector": r.sector,
            "country": r.country,
            "FA0101": value + rng.normal(0, 0.20),
            "FA0102": 0.75 * value + rng.normal(0, 0.35),
            "FA1001": momentum + rng.normal(0, 0.15),
            "FA1002": 0.70 * momentum + rng.normal(0, 0.35),
            "FA2001": quality + rng.normal(0, 0.20),
            "FA2002": -0.60 * quality + rng.normal(0, 0.40),
            "FA3001": growth + rng.normal(0, 0.20),
            "FA3002": 0.70 * growth + rng.normal(0, 0.35),
        })
stocks = pd.DataFrame(rows)

with pd.ExcelWriter(OUT / "factors_and_returns.xlsx", engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
    stocks.to_excel(writer, sheet_name="data", index=False)

master_template = TEMPLATES / "05_factor_master_template.xlsx"
if master_template.exists():
    shutil.copy2(master_template, OUT / "factor_master.xlsx")
else:
    raise FileNotFoundError("05_factor_master_template.xlsx がありません。")

pd.DataFrame(columns=[
    "date", "ISIN", "stock_return", "market_cap", "currency", "sector", "country",
    "FA0101", "FA0102", "FA1001", "FA1002", "FA2001", "FA2002", "FA3001", "FA3002",
]).to_excel(TEMPLATES / "01_factors_and_returns_template.xlsx", sheet_name="data", index=False)

print(f"Stock-only demo files written to {OUT}")
