"""IO utilities for ETF analysis data files."""

from pathlib import Path
import pandas as pd


def load_etf_returns(path: Path, shrcd: int = 73) -> pd.DataFrame:
    """Load CRSP-style ETF returns data.

    Expected columns: date, RET, TICKER (optionally SHRCD for ETF filter).
    Filters by SHRCD when present and coerces RET to numeric.
    """
    df = pd.read_csv(path)
    if "SHRCD" in df.columns:
        df = df[df["SHRCD"] == shrcd]
    df["RET"] = pd.to_numeric(df["RET"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def load_factors(path: Path) -> pd.DataFrame:
    """Load factor CSV with a date column and return a datetime index."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Factors file must include a date column")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")


def load_etf_universe(path: Path) -> pd.DataFrame:
    """Load ETF universe metadata (must include TICKER and CATEGORY)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    needed = {"TICKER", "CATEGORY"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"ETF universe missing columns: {sorted(missing)}")
    return df


def save_etf_universe(df: pd.DataFrame, path: Path) -> None:
    """Save ETF universe metadata to CSV."""
    df.to_csv(path, index=False)
