"""ETF analysis framework package."""

from .config import Paths, Settings
from .io import load_etf_returns, load_factors, load_etf_universe, save_etf_universe
from .etfdb import build_universe, available_filters, fetch_top_by_category
from .prep import select_top_etfs_by_category, build_returns_panel
from .optimization import (
    annualize_stats,
    estimate_factor_model,
    factor_model_cov,
    factor_correlation,
    optimize_min_variance,
    optimize_target_return,
    optimize_max_sharpe,
    optimize_long_only,
)
from .simulation import (
    portfolio_metrics,
    market_vs_idio_risk,
    simulate_portfolios,
    sample_horizon_windows,
    simulate_fixed_portfolio_horizons,
)
from .analysis import top_portfolio_overlap

__all__ = [
    "Paths",
    "Settings",
    "load_etf_returns",
    "load_factors",
    "load_etf_universe",
    "save_etf_universe",
    "build_universe",
    "available_filters",
    "fetch_top_by_category",
    "select_top_etfs_by_category",
    "build_returns_panel",
    "annualize_stats",
    "estimate_factor_model",
    "factor_model_cov",
    "factor_correlation",
    "optimize_min_variance",
    "optimize_target_return",
    "optimize_max_sharpe",
    "optimize_long_only",
    "portfolio_metrics",
    "market_vs_idio_risk",
    "simulate_portfolios",
    "sample_horizon_windows",
    "simulate_fixed_portfolio_horizons",
    "top_portfolio_overlap",
]
