from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import re
import pandas as pd


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())


@dataclass
class SeriesCache:
    root: Path

    @property
    def has_parquet_engine(self) -> bool:
        return importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None

    def path(self, dataset: str, ticker: str) -> Path:
        ext = ".parquet" if self.has_parquet_engine else ".pkl"
        return self.root / safe_name(dataset) / f"{safe_name(ticker)}{ext}"

    def load(self, dataset: str, ticker: str) -> pd.DataFrame:
        path = self.path(dataset, ticker)
        if not path.exists():
            return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_pickle(path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.set_index("date")
        frame.index = pd.to_datetime(frame.index)
        frame.index.name = "date"
        return frame.sort_index()

    def save(self, dataset: str, ticker: str, frame: pd.DataFrame) -> Path:
        path = self.path(dataset, ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = frame.copy().sort_index()
        out.index.name = "date"
        if path.suffix == ".parquet":
            out.reset_index().to_parquet(path, index=False)
        else:
            out.reset_index().to_pickle(path)
        return path

    def merge(self, dataset: str, ticker: str, new_data: pd.DataFrame) -> pd.DataFrame:
        old = self.load(dataset, ticker)
        combined = pd.concat([old, new_data]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        self.save(dataset, ticker, combined)
        return combined
