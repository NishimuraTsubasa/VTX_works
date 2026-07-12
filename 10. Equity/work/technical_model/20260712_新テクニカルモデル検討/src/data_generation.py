from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def load_config(root: Path) -> dict:
    with open(root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_dummy_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(root)
    seed = int(cfg["project"]["random_seed"])
    rng = np.random.default_rng(seed)
    from .excel_io import read_universe, write_market_inputs, write_single_sheet
    universe = read_universe(root)
    dates = pd.bdate_range(cfg["data"]["start_date"], cfg["data"]["end_date"])
    n = len(dates)

    # 3-state Markov regime: neutral, risk-on, risk-off
    transition = np.array([[0.94, 0.04, 0.02], [0.04, 0.94, 0.02], [0.10, 0.08, 0.82]])
    regimes = np.zeros(n, dtype=int)
    regimes[0] = 0
    for t in range(1, n):
        regimes[t] = rng.choice(3, p=transition[regimes[t - 1]])
    mu = np.array([0.00005, 0.00045, -0.00075])
    vol = np.array([0.0085, 0.0070, 0.0210])
    global_ret = mu[regimes] + vol[regimes] * rng.standard_normal(n)

    region_names = universe["region"].unique().tolist()
    region_ret = {}
    for j, region in enumerate(region_names):
        eps = rng.standard_normal(n)
        rr = np.zeros(n)
        for t in range(1, n):
            rr[t] = 0.10 * rr[t - 1] + 0.0038 * eps[t] + (j - 2) * 0.00001
        region_ret[region] = rr

    asset_params = {
        "SPX": (1.00, 0.55, 0.0050), "NDX": (1.18, 0.55, 0.0065), "RTY": (1.10, 0.65, 0.0075),
        "SX5E": (1.02, 0.60, 0.0060), "DAX": (1.08, 0.65, 0.0063), "FTSE": (0.82, 0.55, 0.0052),
        "NKY": (0.95, 0.62, 0.0064), "TPX": (0.90, 0.65, 0.0058),
        "HSI": (1.05, 0.75, 0.0080), "HSCEI": (1.15, 0.80, 0.0090), "KOSPI": (1.10, 0.72, 0.0075),
        "ASX": (0.82, 0.65, 0.0055),
    }

    records = []
    for _, row in universe.iterrows():
        aid = row.asset_id
        beta_g, beta_r, idio_vol = asset_params[aid]
        eps = rng.standard_normal(n)
        idio = np.zeros(n)
        for t in range(1, n):
            idio[t] = 0.06 * idio[t - 1] + idio_vol * eps[t]
        rets = beta_g * global_ret + beta_r * region_ret[row.region] + idio
        # Embed mild momentum and reversal regimes to make model comparison meaningful.
        mom = pd.Series(rets).rolling(20).mean().shift(1).fillna(0).to_numpy()
        rets += np.where(regimes == 1, 0.12 * mom, 0.0)
        rets += np.where(regimes == 0, -0.05 * pd.Series(rets).rolling(5).sum().shift(1).fillna(0).to_numpy(), 0.0)
        close = 100 * np.exp(np.cumsum(rets))
        overnight = 0.25 * rets + 0.002 * rng.standard_normal(n)
        open_px = close / np.exp(overnight)
        intraday_range = np.abs(rets) + 0.004 + 0.004 * (regimes == 2) + 0.002 * rng.random(n)
        high = np.maximum(open_px, close) * (1 + 0.45 * intraday_range)
        low = np.minimum(open_px, close) * np.maximum(0.80, 1 - 0.45 * intraday_range)
        base_vol = rng.lognormal(mean=12.0, sigma=0.25, size=n)
        volume = base_vol * (1 + 13 * np.abs(rets)) * (1 + 0.35 * (regimes == 2))
        oi = np.maximum(10000, np.cumsum(rng.normal(15, 120, n)) + 200000 + 25000 * np.sin(np.arange(n) / 63))
        for k, dt in enumerate(dates):
            records.append((dt, aid, row.region, row.currency, open_px[k], high[k], low[k], close[k], volume[k], oi[k], regimes[k]))

    market = pd.DataFrame(records, columns=["date", "asset_id", "region", "currency", "px_open", "px_high", "px_low", "px_last", "volume", "open_interest", "regime"])

    # Cross-asset synthetic series, all point-in-time market prices/rates.
    global_s = pd.Series(global_ret, index=dates)
    rolling_vol = global_s.rolling(20).std().fillna(global_s.std()) * np.sqrt(252)
    vix = np.clip(11 + 115 * rolling_vol + 9 * (regimes == 2) + rng.normal(0, 1.2, n), 9, 75)
    move = np.clip(55 + 150 * rolling_vol + 22 * (regimes == 2) + rng.normal(0, 3.0, n), 45, 190)

    dxy_ret = -0.18 * global_ret + 0.0017 * rng.standard_normal(n) + 0.0010 * (regimes == 2)
    dxy = 95 * np.exp(np.cumsum(dxy_ret))
    growth = pd.Series(global_ret, index=dates).rolling(30).mean().fillna(0).to_numpy()
    oil_ret = 0.45 * global_ret + 0.25 * growth + 0.010 * rng.standard_normal(n)
    copper_ret = 0.55 * global_ret + 0.40 * growth + 0.008 * rng.standard_normal(n)
    gold_ret = -0.12 * global_ret + 0.004 * (regimes == 2) + 0.005 * rng.standard_normal(n)
    oil = 65 * np.exp(np.cumsum(oil_ret))
    copper = 4 * np.exp(np.cumsum(copper_ret))
    gold = 1300 * np.exp(np.cumsum(gold_ret))
    hy_spread = np.clip(280 + 13 * (vix - 15) + rng.normal(0, 18, n), 180, 1100)

    inflation = np.zeros(n)
    growth_factor = np.zeros(n)
    for t in range(1, n):
        inflation[t] = 0.985 * inflation[t - 1] + 0.03 * oil_ret[t] + rng.normal(0, 0.01)
        growth_factor[t] = 0.97 * growth_factor[t - 1] + 0.10 * global_ret[t] + rng.normal(0, 0.008)
    us2y = np.clip(1.5 + np.cumsum(0.003 * inflation + 0.002 * growth_factor + rng.normal(0, 0.012, n)), 0.0, 7.0)
    us10y = np.clip(2.2 + np.cumsum(0.002 * inflation + 0.003 * growth_factor + rng.normal(0, 0.010, n)), 0.2, 7.0)

    # Local FX levels; signs are chosen only to create varied relationships.
    usdjpy = 105 * np.exp(np.cumsum(0.22 * dxy_ret + 0.0018 * rng.standard_normal(n)))
    eurusd = 1.20 * np.exp(np.cumsum(-0.20 * dxy_ret + 0.0013 * rng.standard_normal(n)))
    gbpusd = 1.35 * np.exp(np.cumsum(-0.18 * dxy_ret + 0.0014 * rng.standard_normal(n)))
    audusd = 0.75 * np.exp(np.cumsum(0.28 * copper_ret - 0.18 * dxy_ret + 0.0016 * rng.standard_normal(n)))
    usdcnh = 6.6 * np.exp(np.cumsum(0.25 * dxy_ret - 0.08 * copper_ret + 0.0012 * rng.standard_normal(n)))
    usdkrw = 1100 * np.exp(np.cumsum(0.22 * dxy_ret - 0.10 * global_ret + 0.0018 * rng.standard_normal(n)))

    cross = pd.DataFrame({
        "date": dates, "vix": vix, "move": move, "dxy": dxy, "us2y": us2y, "us10y": us10y,
        "oil": oil, "copper": copper, "gold": gold, "hy_spread": hy_spread,
        "usdjpy": usdjpy, "eurusd": eurusd, "gbpusd": gbpusd, "audusd": audusd,
        "usdcnh": usdcnh, "usdkrw": usdkrw, "global_regime": regimes,
    })

    write_market_inputs(root, market)
    write_single_sheet(root / "data" / "input" / "Cross_Asset_Data.xlsx", "Cross_Asset_Data", cross)
    return market, cross
