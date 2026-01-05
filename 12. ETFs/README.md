# ETFs Analysis

This project turns the original notebook into a modular Python codebase.

## Quick start

1) Install dependencies:

```bash
pip install pandas numpy statsmodels scipy matplotlib
```

2) Run the analysis:

```bash
python run_analysis.py
```

## Data files

Place these files in the repo root (same folder as `run_analysis.py`):

- `etfs-daily-1980-2024.csv`
- `FF-6factors-1980-2024.csv`
- `etf_universe.csv` (auto-generated if missing)

## What it does

- Builds a tradable ETF universe from ETFdb (top ETFs by category)
- Builds a returns panel from CRSP-style ETF returns
- Simulates diversified portfolios
- Reports common ETFs and asset-class structure among top risk-adjusted portfolios

## Module overview

- `etfs_analysis/io.py`: load returns, factor data, and ETF metadata
- `etfs_analysis/etfdb.py`: fetch ETF universe from ETFdb screener
- `etfs_analysis/prep.py`: select top ETFs and build return panels
- `etfs_analysis/optimization.py`: factor models and portfolio optimizers
- `etfs_analysis/simulation.py`: portfolio simulation and risk decomposition
- `etfs_analysis/analysis.py`: summarize top portfolios and structure

## Example usage (Python)

```python
from etfs_analysis.config import Paths, Settings
from etfs_analysis.io import load_etf_returns, load_factors
from etfs_analysis.etfdb import build_universe
from etfs_analysis.prep import select_top_etfs_by_category, build_returns_panel
from etfs_analysis.simulation import simulate_portfolios
from etfs_analysis.analysis import top_portfolio_overlap

paths = Paths()
settings = Settings()

universe = build_universe(settings.category_fields, top_n=settings.top_n_per_category)
df_etf = load_etf_returns(paths.etf_returns)
top_etfs = select_top_etfs_by_category(universe, top_n=settings.top_n_per_category)
ret_panel = build_returns_panel(df_etf, top_etfs["TICKER"].dropna().unique())

factors = load_factors(paths.factors)
mkt_ret = factors["mktrf"].astype(float) / 100.0 if "mktrf" in factors.columns else None

sim = simulate_portfolios(ret_panel, mkt_ret=mkt_ret, n_portfolios=settings.n_portfolios, etf_counts=settings.etf_counts)
results = top_portfolio_overlap(sim, etf_universe=universe, top_pct=settings.top_pct)
```

## Configuration

Edit `Settings` in `etfs_analysis/config.py` to control:

- `category_fields`
- `top_n_per_category`
- `min_history`
- `n_portfolios`
- `etf_counts`
- `top_pct`
- `etfdb_include_fields` (optional list of ETFdb fields to keep, if available)
