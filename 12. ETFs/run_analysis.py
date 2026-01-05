from pathlib import Path

import pandas as pd

from etfs_analysis.config import Paths, Settings
from etfs_analysis.io import load_etf_returns, load_factors, load_etf_universe, save_etf_universe
from etfs_analysis.etfdb import build_universe
from etfs_analysis.prep import select_top_etfs_by_category, build_returns_panel
from etfs_analysis.simulation import simulate_portfolios
from etfs_analysis.analysis import top_portfolio_overlap


def main():
    paths = Paths()
    settings = Settings()

    if settings.refresh_universe or not paths.universe.exists():
        universe = build_universe(
            settings.category_fields,
            top_n=settings.top_n_per_category,
            include_fields=settings.etfdb_include_fields,
        )
        save_etf_universe(universe, paths.universe)
    else:
        universe = load_etf_universe(paths.universe)

    df_etf = load_etf_returns(paths.etf_returns)

    top_etfs = select_top_etfs_by_category(universe, top_n=settings.top_n_per_category)
    tickers = top_etfs["TICKER"].dropna().unique().tolist()

    ret_panel = build_returns_panel(
        df_etf,
        tickers,
        min_history=settings.min_history,
        fill_method="none",
    )

    factors = None
    if paths.factors.exists():
        factors = load_factors(paths.factors)

    mkt_ret = None
    if factors is not None and "mktrf" in factors.columns:
        mkt_ret = factors["mktrf"].astype(float) / 100.0

    sim = simulate_portfolios(ret_panel, mkt_ret=mkt_ret, n_portfolios=settings.n_portfolios, etf_counts=settings.etf_counts)

    results = top_portfolio_overlap(sim, etf_universe=universe, top_pct=settings.top_pct)

    print("Top tickers in best portfolios:")
    print(results["top_tickers"].to_string())
    print("Average asset mix:")
    if not results["avg_asset_mix"].empty:
        print(results["avg_asset_mix"].to_string())
    else:
        print("(asset mix unavailable; missing CATEGORY_TYPE/asset_class in universe)")
    print("Summary (all vs top):")
    print(results["summary"].to_string())


if __name__ == "__main__":
    main()
