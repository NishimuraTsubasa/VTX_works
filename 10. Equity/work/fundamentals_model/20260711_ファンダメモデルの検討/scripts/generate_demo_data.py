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
indices = {
    "JP_INDEX": {
        "country": "Japan",
        "prefix": "JP",
        "sectors": {"Financials": 0.25, "Industrials": 0.30, "Technology": 0.25, "Consumer": 0.20},
    },
    "US_INDEX": {
        "country": "United States",
        "prefix": "US",
        "sectors": {"Financials": 0.15, "Industrials": 0.20, "Technology": 0.45, "Consumer": 0.20},
    },
    "EU_INDEX": {
        "country": "Europe",
        "prefix": "EU",
        "sectors": {"Financials": 0.25, "Industrials": 0.30, "Technology": 0.15, "Consumer": 0.30},
    },
}

# ファクターユニバースは実構成銘柄リストより意図的に広くしています。
securities: list[dict] = []
for index_name, meta in indices.items():
    sectors = list(meta["sectors"])
    for n in range(48):
        securities.append({
            "index_name": index_name,
            "country": meta["country"],
            "ISIN": f"{meta['prefix']}{n:010d}",
            "sector": sectors[n % len(sectors)],
            "market_cap_base": float(np.exp(rng.normal(24.5, 1.0))),
            "true_index_loading": float(np.clip(rng.normal(1.0, 0.15), 0.5, 1.5)),
        })
sec = pd.DataFrame(securities)
dates = pd.date_range("2017-01-31", periods=72, freq="ME")

latent = {isin: rng.normal(size=4) for isin in sec["ISIN"]}
market_shocks = {idx: rng.normal(0.004, 0.035, len(dates)) for idx in indices}
sector_shocks = {
    (idx, sector): rng.normal(0.0, 0.018, len(dates))
    for idx, meta in indices.items() for sector in meta["sectors"]
}

rows: list[dict] = []
for t, date in enumerate(dates):
    regime = np.sin(t / 12.0)
    for r in sec.itertuples(index=False):
        base = latent[r.ISIN]
        value = 0.65 * base[0] + 0.35 * rng.normal()
        quality = 0.65 * base[1] + 0.35 * rng.normal()
        momentum = 0.50 * base[2] + 0.50 * rng.normal() + 0.15 * regime
        growth = 0.60 * base[3] + 0.40 * rng.normal()
        market = market_shocks[r.index_name][t] * r.true_index_loading
        sector = sector_shocks[(r.index_name, r.sector)][t]
        stock_return = (
            market + sector
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
            "sector": r.sector,
            "country": r.country,
            # 同一潜在特性にノイズを加えた複数FA列
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

# 48銘柄中30銘柄だけを既知の実構成銘柄として入力します。
with pd.ExcelWriter(OUT / "index_constituents.xlsx", engine="xlsxwriter") as writer:
    pd.DataFrame({
        "Description": [
            "One sheet per index. The factor universe may not fully overlap with these constituent lists.",
            "The representative-universe selector can use same-country fallback names.",
        ]
    }).to_excel(writer, sheet_name="README", index=False)
    for index_name, g in sec.groupby("index_name"):
        known = g.sort_values("market_cap_base", ascending=False).head(30)
        known[["ISIN", "sector", "country"]].to_excel(writer, sheet_name=index_name, index=False)

sector_rows = sorted({s for meta in indices.values() for s in meta["sectors"]})
sector_weights = pd.DataFrame({"sector": sector_rows})
for index_name, meta in indices.items():
    sector_weights[index_name] = sector_weights["sector"].map(meta["sectors"]).fillna(0.0)
with pd.ExcelWriter(OUT / "index_sector_weights.xlsx", engine="xlsxwriter") as writer:
    sector_weights.to_excel(writer, sheet_name="sector_weights", index=False)

stock_with_idx = stocks.merge(sec[["ISIN", "index_name", "true_index_loading"]], on="ISIN")
stock_with_idx["proxy_weight"] = stock_with_idx.groupby(["date", "index_name", "sector"])["market_cap"].transform(
    lambda x: x / x.sum()
)
sector_return = (
    stock_with_idx.assign(weighted_return=lambda x: x["stock_return"] * x["proxy_weight"])
    .groupby(["date", "index_name", "sector"], as_index=False)["weighted_return"].sum()
    .rename(columns={"weighted_return": "sector_return"})
)
sector_return = sector_return.merge(
    sector_weights.melt(id_vars="sector", var_name="index_name", value_name="sector_weight"),
    on=["index_name", "sector"], how="left",
)
fut = (
    sector_return.assign(weighted=lambda x: x["sector_return"] * x["sector_weight"])
    .groupby(["date", "index_name"], as_index=False)["weighted"].sum()
    .rename(columns={"weighted": "return"})
)
fut["return"] += rng.normal(0, 0.010, len(fut))
fut_wide = fut.pivot(index="date", columns="index_name", values="return").reset_index()

weekly_dates = pd.date_range(dates.min(), dates.max(), freq="W-FRI")
weekly = pd.DataFrame({"date": weekly_dates})
for col in indices:
    weekly[col] = rng.normal(0.001, 0.025, len(weekly))
with pd.ExcelWriter(OUT / "futures_returns.xlsx", engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
    fut_wide.to_excel(writer, sheet_name="monthly_returns", index=False)
    weekly.to_excel(writer, sheet_name="weekly_returns", index=False)

# factor_masterは整形済みテンプレートを入力へコピーします。
master_template = TEMPLATES / "05_factor_master_template.xlsx"
if master_template.exists():
    shutil.copy2(master_template, OUT / "factor_master.xlsx")
else:
    raise FileNotFoundError("05_factor_master_template.xlsx がありません。")

# 軽量テンプレート。factor_masterは別途整形済みテンプレートを使用します。
pd.DataFrame(columns=[
    "date", "ISIN", "stock_return", "market_cap", "sector", "country",
    "FA0101", "FA0102", "FA1001", "FA1002", "FA2001", "FA2002", "FA3001", "FA3002",
]).to_excel(TEMPLATES / "01_factors_and_returns_template.xlsx", sheet_name="data", index=False)
with pd.ExcelWriter(TEMPLATES / "02_index_constituents_template.xlsx", engine="xlsxwriter") as writer:
    pd.DataFrame(columns=["ISIN", "sector", "country"]).to_excel(writer, sheet_name="INDEX_NAME", index=False)
pd.DataFrame(columns=["sector", "INDEX_NAME"]).to_excel(
    TEMPLATES / "03_index_sector_weights_template.xlsx", sheet_name="sector_weights", index=False
)
with pd.ExcelWriter(TEMPLATES / "04_futures_returns_template.xlsx", engine="xlsxwriter") as writer:
    pd.DataFrame(columns=["date", "INDEX_NAME"]).to_excel(writer, sheet_name="monthly_returns", index=False)
    pd.DataFrame(columns=["date", "INDEX_NAME"]).to_excel(writer, sheet_name="weekly_returns", index=False)

print(f"Demo files written to {OUT}")
