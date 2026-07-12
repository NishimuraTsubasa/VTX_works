from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def _rolling_pct(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    # Percentile of current observation within trailing window; uses only current/past observations.
    return s.rolling(window, min_periods=min_periods).rank(pct=True)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).rolling(window).mean()
    down = (-diff.clip(upper=0)).rolling(window).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_daily_dataset(root: Path) -> pd.DataFrame:
    with open(root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pct_window = int(cfg["data"]["percentile_window"])
    min_hist = int(cfg["data"]["min_percentile_history"])

    from .excel_io import read_market_inputs, read_cross_asset_input, write_single_sheet
    market = read_market_inputs(root)
    cross = read_cross_asset_input(root)
    market = market.sort_values(["asset_id", "date"]).copy()

    pieces = []
    for aid, g in market.groupby("asset_id", sort=False):
        g = g.sort_values("date").copy()
        c = g["px_last"]
        logret = np.log(c).diff()
        for L in [5, 20, 60, 120, 252]:
            g[f"ret{L}"] = c.pct_change(L)
        for L in [20, 60, 120]:
            ma = c.rolling(L).mean()
            g[f"ma_gap{L}"] = c / ma - 1
        ma60 = c.rolling(60).mean()
        g["ma_slope60"] = ma60.pct_change(20)
        g["breakout120"] = c / c.rolling(120).max() - 1
        g["drawdown60"] = c / c.rolling(60).max() - 1
        g["drawdown252"] = c / c.rolling(252).max() - 1
        g["rv20"] = logret.rolling(20).std() * np.sqrt(252)
        g["rv60"] = logret.rolling(60).std() * np.sqrt(252)
        g["vol_ratio"] = g["rv20"] / g["rv60"]
        prev_close = c.shift(1)
        tr = pd.concat([(g.px_high - g.px_low), (g.px_high - prev_close).abs(), (g.px_low - prev_close).abs()], axis=1).max(axis=1)
        g["atr_pct"] = tr.rolling(20).mean() / c
        g["volume_ratio"] = g.volume / g.volume.rolling(20).mean()
        g["oi_change20"] = g.open_interest.pct_change(20)
        g["rsi14"] = _rsi(c, 14)
        ma20 = c.rolling(20).mean()
        sd20 = c.rolling(20).std()
        g["bb_position"] = (c - ma20) / (2 * sd20.replace(0, np.nan))
        pieces.append(g)
    df = pd.concat(pieces, ignore_index=True)

    # Cross-sectional return ranks and region-relative returns.
    for col in ["ret20", "ret60", "ret120"]:
        df[f"{col}_cs"] = df.groupby("date")[col].rank(pct=True)
    df["global_ret60_mean"] = df.groupby("date")["ret60"].transform("mean")
    df["rel_global60"] = df["ret60"] - df["global_ret60_mean"]
    df["region_ret60_mean"] = df.groupby(["date", "region"])["ret60"].transform("mean")
    df["rel_region60"] = df["ret60"] - df["region_ret60_mean"]

    # Lag cross-asset variables one business day to be conservative on time-zone alignment.
    cross = cross.sort_values("date").copy()
    level_cols = [c for c in cross.columns if c not in {"date", "global_regime"}]
    for col in level_cols:
        cross[f"{col}_ret20"] = cross[col].pct_change(20)
    cross["us10y_chg20"] = cross.us10y.diff(20)
    cross["us2y_chg20"] = cross.us2y.diff(20)
    cross["curve"] = cross.us10y - cross.us2y
    cross["curve_chg20"] = cross.curve.diff(20)
    cross["global_eq_ret20"] = market.pivot(index="date", columns="asset_id", values="px_last").pct_change(20).mean(axis=1).reindex(cross.date).to_numpy()
    lag_cols = [c for c in cross.columns if c not in {"date", "global_regime"}]
    cross[lag_cols] = cross[lag_cols].shift(1)
    df = df.merge(cross, on="date", how="left")

    # Market-specific local FX support: higher is intended as more equity-supportive before percentile transform.
    fx_map = {
        "SPX": ("dxy_ret20", -1), "NDX": ("dxy_ret20", -1), "RTY": ("dxy_ret20", -0.3),
        "SX5E": ("eurusd_ret20", -1), "DAX": ("eurusd_ret20", -1), "FTSE": ("gbpusd_ret20", -1),
        "NKY": ("usdjpy_ret20", 1), "TPX": ("usdjpy_ret20", 0.7),
        "HSI": ("usdcnh_ret20", -1), "HSCEI": ("usdcnh_ret20", -1), "KOSPI": ("usdkrw_ret20", -0.5),
        "ASX": ("audusd_ret20", 0.5),
    }
    df["local_fx_support_raw"] = np.nan
    for aid, (col, sign) in fx_map.items():
        df.loc[df.asset_id == aid, "local_fx_support_raw"] = sign * df.loc[df.asset_id == aid, col]

    signal_base = [
        "ret5", "ret20", "ret60", "ret120", "ma_gap20", "ma_gap60", "ma_gap120", "ma_slope60",
        "breakout120", "drawdown60", "drawdown252", "rv20", "vol_ratio", "atr_pct", "volume_ratio",
        "oi_change20", "rsi14", "bb_position", "rel_global60", "rel_region60", "vix", "move", "hy_spread",
        "dxy_ret20", "us10y_chg20", "us2y_chg20", "curve_chg20", "oil_ret20", "copper_ret20", "gold_ret20",
        "global_eq_ret20", "local_fx_support_raw",
    ]
    # Time-series percentile by asset for local features; global series percentile is identical across assets.
    local_cols = set(signal_base[:20] + ["local_fx_support_raw"])
    for col in signal_base:
        if col in local_cols:
            df[f"{col}_pct"] = df.groupby("asset_id", group_keys=False)[col].apply(lambda s: _rolling_pct(s, pct_window, min_hist))
        else:
            # avoid duplicate asset rows by ranking global series once and mapping back
            unique = df[["date", col]].drop_duplicates("date").sort_values("date")
            unique[f"{col}_pct"] = _rolling_pct(unique[col], pct_window, min_hist)
            df = df.merge(unique[["date", f"{col}_pct"]], on="date", how="left")

    # Evidence scores on 0..1 scale.
    df["evidence_persistence"] = (
        0.20 * df.ret20_pct + 0.25 * df.ret60_pct + 0.15 * df.ret120_pct +
        0.15 * df.ma_gap60_pct + 0.10 * df.ma_slope60_pct + 0.15 * df.breakout120_pct
    )
    overbought = 0.30 * df.ret5_pct + 0.20 * df.ret20_pct + 0.30 * df.rsi14_pct + 0.20 * df.bb_position_pct
    oversold = 0.30 * (1 - df.ret5_pct) + 0.20 * (1 - df.ret20_pct) + 0.30 * (1 - df.rsi14_pct) + 0.20 * (1 - df.drawdown60_pct)
    df["evidence_correction"] = (0.5 + 0.5 * (oversold - overbought)).clip(0, 1)
    df["evidence_volatility_support"] = (
        0.30 * (1 - df.rv20_pct) + 0.20 * (1 - df.vol_ratio_pct) + 0.15 * (1 - df.atr_pct_pct) +
        0.25 * (1 - df.vix_pct) + 0.10 * (1 - df.move_pct)
    )
    participation = (0.50 * df.volume_ratio_pct + 0.50 * df.oi_change20_pct).clip(0, 1)
    direction = 2 * df.ret20_pct - 1
    df["evidence_flow"] = (0.5 + 0.5 * direction * participation).clip(0, 1)
    df["evidence_relative_strength"] = (
        0.30 * df.ret20_cs + 0.35 * df.ret60_cs + 0.20 * df.ret120_cs + 0.15 * df.rel_global60_pct
    )
    df["evidence_intermarket"] = (
        0.25 * df.global_eq_ret20_pct + 0.25 * (1 - df.vix_pct) + 0.20 * (1 - df.hy_spread_pct) +
        0.15 * (1 - df.dxy_ret20_pct) + 0.15 * df.copper_ret20_pct
    )
    df["evidence_macro_market"] = (
        0.25 * (1 - df.us10y_chg20_pct) + 0.15 * (1 - df.us2y_chg20_pct) +
        0.20 * df.local_fx_support_raw_pct + 0.20 * df.copper_ret20_pct + 0.10 * df.oil_ret20_pct +
        0.10 * (1 - df.gold_ret20_pct)
    )

    # Calendar signals.
    df["month"] = df.date.dt.month
    df["quarter_end"] = df.date.dt.month.isin([3, 6, 9, 12]).astype(int)
    df["turn_of_month"] = ((df.date.dt.day <= 3) | (df.date.dt.day >= 27)).astype(int)

    if bool(cfg.get("outputs", {}).get("write_full_daily_signals", False)):
        write_single_sheet(root / "data" / "processed" / "Daily_Signals.xlsx", "Daily_Signals", df)
    return df


def build_monthly_dataset(root: Path, daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    monthly_dates = daily.groupby(daily.date.dt.to_period("M"))["date"].max().sort_values().tolist()
    m = daily[daily.date.isin(monthly_dates)].copy().sort_values(["date", "asset_id"])
    # Next monthly close-to-close return and cross-sectional rank target.
    m["target_return"] = m.groupby("asset_id")["px_last"].shift(-1) / m["px_last"] - 1
    m["target_rank"] = m.groupby("date")["target_return"].rank(pct=True)
    m["target_scaled"] = 2 * m["target_rank"] - 1

    evidence_cols = [
        "evidence_persistence", "evidence_correction", "evidence_volatility_support", "evidence_flow",
        "evidence_relative_strength", "evidence_intermarket", "evidence_macro_market",
    ]
    signal_cols = [
        "ret5_pct", "ret20_pct", "ret60_pct", "ret120_pct", "ma_gap20_pct", "ma_gap60_pct", "ma_gap120_pct",
        "ma_slope60_pct", "breakout120_pct", "drawdown60_pct", "drawdown252_pct", "rv20_pct", "vol_ratio_pct",
        "atr_pct_pct", "volume_ratio_pct", "oi_change20_pct", "rsi14_pct", "bb_position_pct", "ret20_cs",
        "ret60_cs", "ret120_cs", "rel_global60_pct", "rel_region60_pct", "vix_pct", "move_pct", "hy_spread_pct",
        "dxy_ret20_pct", "us10y_chg20_pct", "us2y_chg20_pct", "curve_chg20_pct", "oil_ret20_pct",
        "copper_ret20_pct", "gold_ret20_pct", "global_eq_ret20_pct", "local_fx_support_raw_pct",
        "turn_of_month", "quarter_end",
    ]
    write_single_sheet(root / "data" / "processed" / "Monthly_Model_Dataset.xlsx", "Monthly_Model", m)
    return m, signal_cols, evidence_cols
