from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "input"


def generate(seed: int = 42, n_stocks: int = 100, n_months: int = 48) -> None:
    """v0.12.2動作確認用の合成factors_and_returns.xlsxを再生成する。"""
    rng = np.random.default_rng(seed)
    countries = ["US", "UK", "Germany", "Japan", "Canada"]
    sectors = ["Financials", "Industrials", "Consumer", "Healthcare", "Technology"]
    currencies = {"US": "USD", "UK": "GBP", "Germany": "EUR", "Japan": "JPY", "Canada": "CAD"}
    dates = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    factor_codes = ["FA0101", "FA0102", "FA1001", "FA2001", "FA3001", "FA4001"]

    info = []
    for i in range(n_stocks):
        country = countries[i % len(countries)]
        sector = sectors[(i // len(countries)) % len(sectors)]
        info.append((f"ZZ{i:010d}", country, sector, currencies[country], float(np.exp(rng.normal(23.5, 1.0)))))

    state = np.zeros((n_stocks, len(factor_codes)))
    prev = state.copy()
    rows = []
    for t, date in enumerate(dates):
        state = 0.65 * state + 0.75 * rng.normal(size=state.shape)
        for i, (isin, country, sector, currency, base_cap) in enumerate(info):
            sector_signal = 0.0
            if sector == "Financials":
                sector_signal = 0.004 * prev[i, 0] - 0.002 * prev[i, 3]
            elif sector == "Technology":
                sector_signal = 0.004 * prev[i, 2] + 0.002 * prev[i, 4]
            elif sector == "Industrials":
                sector_signal = 0.003 * prev[i, 2] + 0.002 * prev[i, 0]
            elif sector == "Healthcare":
                sector_signal = -0.002 * prev[i, 3] + 0.002 * prev[i, 1]
            stock_return = (
                0.004 * prev[i, 0]
                + 0.003 * prev[i, 2]
                - 0.0025 * prev[i, 3]
                + sector_signal
                + rng.normal(0, 0.04)
            )
            factors = state[i].astype(object)
            factors[rng.random(len(factors)) < 0.04] = np.nan
            rows.append({
                "date": date,
                "ISIN": isin,
                "stock_return": stock_return,
                "market_cap": base_cap * np.exp(rng.normal(0, 0.05) + 0.001 * t),
                "currency": currency,
                "sector": sector,
                "country": country,
                **dict(zip(factor_codes, factors)),
            })
        prev = state.copy()

    data = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "factors_and_returns.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        pd.DataFrame({
            "Item": ["Purpose", "Grain", "Return", "Missing"],
            "Description": [
                "v0.12.2動作確認用の合成データ。実データへ置換してください。",
                "1行 = date x ISIN",
                "stock_returnは当該月リターン。Configで翌月へシフト。",
                "FA欠損はNaN。0埋めしない。",
            ],
        }).to_excel(writer, sheet_name="README", index=False)
        pd.DataFrame([
            ["date", "Date", "ファクター観測時点"],
            ["ISIN", "Text", "銘柄キー"],
            ["stock_return", "Decimal", "当該月リターン"],
            ["market_cap", "Decimal", "時価総額"],
            ["currency", "Text", "通貨"],
            ["sector", "Text", "セクター"],
            ["country", "Text", "国"],
            ["FAxxxx", "Decimal", "ファクター値"],
        ], columns=["Column", "Type", "Description"]).to_excel(writer, sheet_name="Column_Dictionary", index=False)
        data.to_excel(writer, sheet_name="data", index=False)
    print(f"Generated: {path}")


if __name__ == "__main__":
    generate()
