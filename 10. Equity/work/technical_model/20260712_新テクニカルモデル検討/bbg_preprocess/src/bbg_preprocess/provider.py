from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import pandas as pd


class HistoricalDataProvider(ABC):
    @abstractmethod
    def fetch_history(
        self,
        ticker: str,
        fields: list[str],
        start_date: str,
        end_date: str,
        periodicity: str = "DAILY",
        overrides: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by date, columns named as requested Bloomberg fields."""


class XbbgProvider(HistoricalDataProvider):
    def __init__(self) -> None:
        try:
            from xbbg import blp  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "xbbg is not installed. Install Bloomberg's blpapi and xbbg in a Bloomberg-connected environment."
            ) from exc
        self._blp = blp

    def fetch_history(
        self,
        ticker: str,
        fields: list[str],
        start_date: str,
        end_date: str,
        periodicity: str = "DAILY",
        overrides: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        kwargs: dict[str, Any] = {"Per": periodicity}
        if overrides:
            kwargs.update(overrides)
        raw = self._blp.bdh(
            tickers=ticker,
            flds=fields,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=fields, index=pd.DatetimeIndex([], name="date"))
        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            # xbbg commonly returns (ticker, field). A one-ticker request is used here.
            frame.columns = [str(c[-1]).upper() for c in frame.columns]
        else:
            frame.columns = [str(c).upper() for c in frame.columns]
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        return frame.reindex(columns=[f.upper() for f in fields])


class MockProvider(HistoricalDataProvider):
    """Provider that routes to the synthetic inputs already distributed with this project."""

    def __init__(self, project_root: Path, futures_lookup: dict[str, str], cross_lookup: dict[str, str]):
        self.project_root = project_root
        self.futures_lookup = futures_lookup
        self.cross_lookup = cross_lookup

    def fetch_history(
        self,
        ticker: str,
        fields: list[str],
        start_date: str,
        end_date: str,
        periodicity: str = "DAILY",
        overrides: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        inv_fut = {v: k for k, v in self.futures_lookup.items()}
        if ticker in inv_fut:
            aid = inv_fut[ticker]
            path = self.project_root / "data" / "input" / "market" / f"Market_{aid}.xlsx"
            src = pd.read_excel(path, sheet_name="Market_Data", parse_dates=["date"])
            mapping = {
                "PX_OPEN": "px_open", "PX_HIGH": "px_high", "PX_LOW": "px_low",
                "PX_LAST": "px_last", "PX_VOLUME": "volume", "OPEN_INT": "open_interest",
            }
            src = src[(src.date >= start) & (src.date <= end)].set_index("date")
            out = pd.DataFrame(index=src.index)
            for field in fields:
                col = mapping.get(field.upper())
                out[field.upper()] = src[col] if col in src.columns else pd.NA
            return out

        inv_cross = {v: k for k, v in self.cross_lookup.items()}
        if ticker in inv_cross:
            output_col = inv_cross[ticker]
            path = self.project_root / "data" / "input" / "Cross_Asset_Data.xlsx"
            src = pd.read_excel(path, sheet_name="Cross_Asset_Data", parse_dates=["date"])
            src = src[(src.date >= start) & (src.date <= end)].set_index("date")
            out = pd.DataFrame(index=src.index)
            for field in fields:
                out[field.upper()] = src[output_col] if output_col in src.columns else pd.NA
            return out
        return pd.DataFrame(columns=fields, index=pd.DatetimeIndex([], name="date"))
