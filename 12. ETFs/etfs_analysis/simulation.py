"""Simulation utilities for portfolio risk/return analysis."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def portfolio_metrics(ret, rf=0.0, periods_per_year=252):
    """Compute annualized return/vol, Sharpe, and max drawdown."""
    mean = ret.mean() * periods_per_year
    vol = ret.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = np.nan if vol == 0 else (mean - rf) / vol
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    drawdown = (cum / peak) - 1
    max_dd = drawdown.min()
    return {"ann_return": mean, "ann_vol": vol, "sharpe": sharpe, "max_dd": max_dd}


def market_vs_idio_risk(port_ret, mkt_ret):
    """Decompose portfolio variance into market and idiosyncratic components."""
    aligned = pd.concat([port_ret, mkt_ret], axis=1).dropna()
    if aligned.empty:
        return {"beta": np.nan, "mkt_var": np.nan, "idio_var": np.nan, "idio_share": np.nan}
    y = aligned.iloc[:, 0]
    X = sm.add_constant(aligned.iloc[:, 1])
    res = sm.OLS(y, X).fit()
    beta = res.params.iloc[1]
    mkt_var = (beta ** 2) * aligned.iloc[:, 1].var(ddof=0)
    idio_var = res.resid.var(ddof=0)
    total = y.var(ddof=0)
    idio_share = np.nan if total == 0 else idio_var / total
    return {"beta": beta, "mkt_var": mkt_var, "idio_var": idio_var, "idio_share": idio_share}


def simulate_portfolios(returns, mkt_ret=None, n_portfolios=500, etf_counts=(5, 10, 20), random_state=42):
    """Simulate equal-weight portfolios across ETF counts."""
    rng = np.random.default_rng(random_state)
    tickers = list(returns.columns)
    results = []
    for k in etf_counts:
        if k > len(tickers):
            continue
        for _ in range(n_portfolios):
            picks = rng.choice(tickers, size=k, replace=False)
            port_ret = returns[picks].mean(axis=1)
            metrics = portfolio_metrics(port_ret)
            risk = {}
            if mkt_ret is not None:
                risk = market_vs_idio_risk(port_ret, mkt_ret)
            results.append({"n_etfs": k, "tickers": ",".join(picks), **metrics, **risk})
    return pd.DataFrame(results)


def sample_horizon_windows(returns, years, n_samples=100, random_state=42):
    """Sample rolling windows of a fixed horizon (years) from returns."""
    rng = np.random.default_rng(random_state)
    dates = returns.index
    if dates.empty:
        return []
    end_max = dates.max()
    start_min = dates.min()
    horizon = pd.DateOffset(years=years)
    latest_start = end_max - horizon
    valid_starts = dates[dates <= latest_start]
    if valid_starts.empty:
        return []
    starts = rng.choice(valid_starts, size=n_samples, replace=True)
    return [(pd.Timestamp(s), pd.Timestamp(s) + horizon) for s in starts]


def simulate_fixed_portfolio_horizons(returns, tickers, years, n_samples=100, random_state=42):
    """Simulate a fixed ticker set across random horizon windows."""
    window_list = sample_horizon_windows(returns, years, n_samples=n_samples, random_state=random_state)
    if not window_list:
        return pd.DataFrame()
    panel = returns.loc[:, returns.columns.intersection(tickers)]
    results = []
    for start, end in window_list:
        window = panel.loc[(panel.index >= start) & (panel.index <= end)]
        if window.empty:
            continue
        port_ret = window.mean(axis=1)
        metrics = portfolio_metrics(port_ret)
        metrics.update({"start": start, "end": end})
        results.append(metrics)
    return pd.DataFrame(results)
