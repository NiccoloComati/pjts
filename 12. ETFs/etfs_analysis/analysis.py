"""Higher-level analysis helpers for simulation outputs."""

import pandas as pd


def top_portfolio_overlap(sim, etf_universe=None, top_pct=0.05, top_n=20):
    """Summarize overlap and structure among top risk-adjusted portfolios."""
    sim_sorted = sim.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)
    n = max(1, int(len(sim_sorted) * top_pct))
    top = sim_sorted.head(n).copy()
    top["ticker_list"] = top["tickers"].str.split(",")

    all_tickers = pd.Series([t for lst in top["ticker_list"] for t in lst])
    ticker_counts = all_tickers.value_counts().rename("count")
    ticker_freq = (ticker_counts / n).rename("freq_top")
    top_ticker_summary = pd.concat([ticker_counts, ticker_freq], axis=1).head(top_n)

    avg_mix = pd.Series(dtype=float)
    if etf_universe is not None and "CATEGORY_TYPE" in etf_universe.columns:
        tmp = etf_universe.copy()
        tmp = tmp[tmp["CATEGORY_TYPE"].str.lower() == "asset_class"]
        asset_map = tmp[["TICKER", "CATEGORY"]].drop_duplicates("TICKER").set_index("TICKER")["CATEGORY"]

        def portfolio_asset_mix(ticker_list):
            classes = pd.Series(ticker_list).map(asset_map).fillna("Unknown")
            return classes.value_counts(normalize=True)

        mixes = top["ticker_list"].apply(portfolio_asset_mix)
        avg_mix = mixes.fillna(0).mean().sort_values(ascending=False)

    metric_cols = ["ann_return", "ann_vol", "sharpe", "max_dd", "idio_share"]
    summary_all = sim[metric_cols].mean(numeric_only=True)
    summary_top = top[metric_cols].mean(numeric_only=True)
    summary = pd.concat([summary_all.rename("all"), summary_top.rename("top")], axis=1)

    return {
        "top": top,
        "top_tickers": top_ticker_summary,
        "avg_asset_mix": avg_mix,
        "summary": summary,
    }
