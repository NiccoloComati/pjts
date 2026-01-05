"""Configuration defaults for ETF analysis runs."""

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR


@dataclass(frozen=True)
class Paths:
    etf_returns: Path = DATA_DIR / "etfs-daily-1980-2024.csv"
    factors: Path = DATA_DIR / "FF-6factors-1980-2024.csv"
    universe: Path = DATA_DIR / "etf_universe.csv"


@dataclass
class Settings:
    refresh_universe: bool = True
    top_n_per_category: int = 5
    category_fields: tuple = ("asset_class", "sizes", "investment_styles")
    etfdb_include_fields: tuple | None = None
    min_history: int = 252
    n_portfolios: int = 300
    etf_counts: tuple = (5, 10, 20)
    top_pct: float = 0.05
