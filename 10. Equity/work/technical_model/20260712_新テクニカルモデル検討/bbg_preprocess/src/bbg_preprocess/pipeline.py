from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import sleep
from typing import Any
import traceback
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from .cache import SeriesCache
from .config_loader import PreprocessConfig, load_config, parse_config_date
from .excel_output import write_frame, write_multi_sheet
from .provider import HistoricalDataProvider, MockProvider, XbbgProvider


@dataclass
class QueryResult:
    dataset: str
    identifier: str
    ticker: str
    request_start: date | None
    request_end: date | None
    cache_start: date | None
    cache_end: date | None
    rows_before: int
    rows_downloaded: int
    rows_after: int
    status: str
    message: str


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def _settings_bool(settings: dict[str, Any], key: str, default: bool) -> bool:
    return _to_bool(settings.get(key, default))


def _resolve_project_root(config_path: Path) -> Path:
    # .../project/bbg_preprocess/config/BBG_Config.xlsx -> project root
    return config_path.resolve().parents[2]


def _make_provider(cfg: PreprocessConfig, root: Path) -> HistoricalDataProvider:
    provider = str(cfg.settings.get("provider", "mock")).strip().lower()
    futures_lookup = dict(zip(cfg.futures.asset_id.astype(str), cfg.futures.bbg_ticker.astype(str)))
    cross_lookup = dict(zip(cfg.cross_asset.output_column.astype(str), cfg.cross_asset.bbg_ticker.astype(str)))
    if provider == "mock":
        return MockProvider(root, futures_lookup=futures_lookup, cross_lookup=cross_lookup)
    if provider == "xbbg":
        return XbbgProvider()
    raise ValueError(f"Unknown provider: {provider}")


def _overrides_for(cfg: PreprocessConfig, ticker: str) -> dict[str, Any]:
    if cfg.overrides.empty:
        return {}
    rows = cfg.overrides[(cfg.overrides.active == True) & (cfg.overrides.bbg_ticker.astype(str) == ticker)]
    return {str(r.override_field): r.override_value for _, r in rows.iterrows() if pd.notna(r.override_field)}


def _request_window(
    cached: pd.DataFrame,
    global_start: date,
    global_end: date,
    overlap_days: int,
    run_mode: str,
    force_refresh: bool,
) -> tuple[date | None, date | None]:
    if run_mode == "full_refresh" or force_refresh or cached.empty:
        return global_start, global_end
    last = cached.index.max().date()
    start = (pd.Timestamp(last) - BDay(overlap_days)).date()
    start = max(start, global_start)
    if start > global_end:
        return None, None
    return start, global_end


def _fetch_with_retry(
    provider: HistoricalDataProvider,
    ticker: str,
    fields: list[str],
    start: date,
    end: date,
    periodicity: str,
    overrides: dict[str, Any],
    retries: int,
    wait_seconds: float,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return provider.fetch_history(
                ticker=ticker,
                fields=fields,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                periodicity=periodicity,
                overrides=overrides,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                sleep(wait_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


def _quality_row(dataset: str, identifier: str, ticker: str, frame: pd.DataFrame, required_fields: list[str], end_date: date, stale_warn: int, move_warn: float) -> dict[str, Any]:
    if frame.empty:
        return {
            "dataset": dataset, "identifier": identifier, "ticker": ticker, "row_count": 0,
            "start_date": None, "end_date": None, "missing_required_cells": None,
            "duplicate_dates": 0, "stale_business_days": None, "large_move_count": None,
            "quality_status": "INVALID",
        }
    dup = int(frame.index.duplicated().sum())
    missing = int(frame.reindex(columns=required_fields).isna().sum().sum()) if required_fields else 0
    last_date = frame.index.max().date()
    stale = int(np.busday_count(last_date, end_date)) if last_date <= end_date else 0
    large_moves = None
    if "PX_LAST" in frame.columns:
        large_moves = int((frame["PX_LAST"].pct_change().abs() > move_warn).sum())
    status = "OK"
    if missing > 0 or dup > 0:
        status = "WARNING"
    if stale > stale_warn:
        status = "WARNING"
    return {
        "dataset": dataset, "identifier": identifier, "ticker": ticker,
        "row_count": len(frame), "start_date": frame.index.min().date(), "end_date": last_date,
        "missing_required_cells": missing, "duplicate_dates": dup,
        "stale_business_days": stale, "large_move_count": large_moves, "quality_status": status,
    }


def run(config_path: Path) -> dict[str, Path]:
    cfg = load_config(config_path)
    root = _resolve_project_root(config_path)
    settings = cfg.settings
    run_control = cfg.run_control
    provider = _make_provider(cfg, root)

    global_start = parse_config_date(run_control.get("run_start_date"), parse_config_date(settings.get("default_start_date")))
    global_end = parse_config_date(run_control.get("run_end_date"), parse_config_date(settings.get("default_end_date"), date.today()))
    run_mode = str(settings.get("run_mode", "incremental")).strip().lower()
    overlap = int(settings.get("overlap_business_days", 10))
    periodicity = str(settings.get("periodicity", "DAILY"))
    retries = int(settings.get("retry_count", 3))
    wait_seconds = float(settings.get("retry_wait_seconds", 2))
    stale_warn = int(settings.get("stale_warning_business_days", 3))
    move_warn = float(settings.get("large_move_warning", 0.15))
    fail_required = _settings_bool(settings, "fail_on_required_missing", True)
    cache = SeriesCache(root / str(settings.get("cache_root", "data/bbg_cache")))
    force = {x.strip() for x in str(run_control.get("force_refresh_tickers", "")).split(",") if x.strip()}

    query_log: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    futures_cache: dict[str, pd.DataFrame] = {}
    cross_cache: dict[str, pd.DataFrame] = {}

    fields_frame = cfg.futures_fields[cfg.futures_fields.active == True].copy()
    bbg_fields = fields_frame.bbg_field.astype(str).str.upper().tolist()
    required_fields = fields_frame.loc[fields_frame.required == True, "bbg_field"].astype(str).str.upper().tolist()

    active_futures = cfg.futures[cfg.futures.active == True].copy()
    for _, row in active_futures.iterrows():
        aid, ticker = str(row.asset_id), str(row.bbg_ticker)
        cached = cache.load("futures", ticker)
        start, end = _request_window(cached, global_start, global_end, overlap, run_mode, ticker in force)
        before = len(cached)
        downloaded = pd.DataFrame()
        status, message = "CACHE_ONLY", "Cache already covers requested period"
        if start is not None and run_mode != "dry_run":
            try:
                downloaded = _fetch_with_retry(provider, ticker, bbg_fields, start, end, periodicity, _overrides_for(cfg, ticker), retries, wait_seconds)
                cached = cache.merge("futures", ticker, downloaded)
                status, message = "DOWNLOADED", "Incremental range merged into cache"
            except Exception as exc:  # noqa: BLE001
                status, message = "ERROR", f"{type(exc).__name__}: {exc}"
                if _to_bool(row.required) and fail_required:
                    raise
        elif start is not None and run_mode == "dry_run":
            status, message = "DRY_RUN", "No Bloomberg request was sent"
        futures_cache[aid] = cached
        query_log.append(QueryResult("futures", aid, ticker, start, end, cached.index.min().date() if not cached.empty else None, cached.index.max().date() if not cached.empty else None, before, len(downloaded), len(cached), status, message).__dict__)
        quality.append(_quality_row("futures", aid, ticker, cached, required_fields, global_end, stale_warn, move_warn))

    active_cross = cfg.cross_asset[cfg.cross_asset.active == True].copy()
    for _, row in active_cross.iterrows():
        sid, ticker, field = str(row.series_id), str(row.bbg_ticker), str(row.bbg_field).upper()
        cached = cache.load("cross_asset", ticker)
        start, end = _request_window(cached, global_start, global_end, overlap, run_mode, ticker in force)
        before = len(cached)
        downloaded = pd.DataFrame()
        status, message = "CACHE_ONLY", "Cache already covers requested period"
        if start is not None and run_mode != "dry_run":
            try:
                downloaded = _fetch_with_retry(provider, ticker, [field], start, end, periodicity, _overrides_for(cfg, ticker), retries, wait_seconds)
                cached = cache.merge("cross_asset", ticker, downloaded)
                status, message = "DOWNLOADED", "Incremental range merged into cache"
            except Exception as exc:  # noqa: BLE001
                status, message = "ERROR", f"{type(exc).__name__}: {exc}"
                if _to_bool(row.required) and fail_required:
                    raise
        elif start is not None and run_mode == "dry_run":
            status, message = "DRY_RUN", "No Bloomberg request was sent"
        cross_cache[str(row.output_column)] = cached[[field]].rename(columns={field: str(row.output_column)}) if field in cached.columns else pd.DataFrame(index=cached.index, columns=[str(row.output_column)])
        query_log.append(QueryResult("cross_asset", sid, ticker, start, end, cached.index.min().date() if not cached.empty else None, cached.index.max().date() if not cached.empty else None, before, len(downloaded), len(cached), status, message).__dict__)
        quality.append(_quality_row("cross_asset", sid, ticker, cached, [field] if _to_bool(row.required) else [], global_end, stale_warn, move_warn))

    # Export model inputs only when not dry-run and not explicitly skipped.
    outputs: dict[str, Path] = {}
    skip_export = _to_bool(run_control.get("skip_model_input_export", False))
    if run_mode != "dry_run" and not skip_export:
        universe_out = active_futures[["asset_id", "region", "currency", "bbg_ticker", "active"]].copy()
        universe_out["active"] = True
        path = root / "data" / "input" / "Universe_Master.xlsx"
        write_frame(path, "Universe", universe_out)
        outputs["universe"] = path

        market_dir = root / "data" / "input" / "market"
        for _, row in active_futures.iterrows():
            aid = str(row.asset_id)
            frame = futures_cache[aid].copy().reset_index()
            rename = dict(zip(fields_frame.bbg_field.astype(str).str.upper(), fields_frame.output_column.astype(str)))
            frame = frame.rename(columns=rename)
            frame["asset_id"] = aid
            frame["region"] = row.region
            frame["currency"] = row.currency
            cols = ["date", "asset_id", "region", "currency"] + [c for c in rename.values() if c in frame.columns]
            frame = frame[cols].sort_values("date")
            path = market_dir / f"Market_{aid}.xlsx"
            write_frame(path, "Market_Data", frame)
        outputs["market_dir"] = market_dir

        cross_frame = None
        for _, frame in cross_cache.items():
            cross_frame = frame if cross_frame is None else cross_frame.join(frame, how="outer")
        if cross_frame is None:
            cross_frame = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        cross_frame = cross_frame.sort_index().reset_index()
        path = root / "data" / "input" / "Cross_Asset_Data.xlsx"
        write_frame(path, "Cross_Asset_Data", cross_frame)
        outputs["cross_asset"] = path

    log_dir = root / "bbg_preprocess" / "logs"
    query_df = pd.DataFrame(query_log)
    quality_df = pd.DataFrame(quality)
    config_snapshot = pd.DataFrame({
        "key": ["provider", "run_mode", "global_start", "global_end", "overlap_business_days", "cache_backend"],
        "value": [settings.get("provider"), run_mode, global_start, global_end, overlap, settings.get("cache_backend")],
    })
    qlog_path = log_dir / "BBG_Query_Log.xlsx"
    write_multi_sheet(qlog_path, {"Query_Log": query_df, "Run_Settings": config_snapshot})
    quality_path = log_dir / "BBG_Data_Quality.xlsx"
    write_multi_sheet(quality_path, {"Series_Coverage": quality_df, "Query_Summary": query_df})
    outputs["query_log"] = qlog_path
    outputs["quality_report"] = quality_path
    return outputs
