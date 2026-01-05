"""Portfolio optimization and factor-model utilities."""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def annualize_stats(returns, periods_per_year=252):
    """Return annualized mean and covariance from daily returns."""
    mu = returns.mean() * periods_per_year
    cov = returns.cov(ddof=0) * periods_per_year
    return mu, cov


def _maybe_scale_factors(factors):
    """Scale factor returns to decimals if inputs look like percent values."""
    med = factors.abs().median().median()
    if pd.notna(med) and med > 1:
        return factors / 100.0
    return factors


def estimate_factor_model(returns, factors, factor_cols=None):
    """Estimate factor betas and idiosyncratic variances via OLS."""
    if factor_cols is None:
        factor_cols = [c for c in factors.columns if c.lower() not in ("rf",)]
    fac = factors[factor_cols].copy()
    fac = _maybe_scale_factors(fac)
    aligned = returns.join(fac, how="inner")
    fac = aligned[factor_cols]
    y = aligned[returns.columns]
    X = sm.add_constant(fac)

    betas = pd.DataFrame(index=returns.columns, columns=factor_cols, dtype=float)
    idio_var = pd.Series(index=returns.columns, dtype=float)

    for col in y.columns:
        res = sm.OLS(y[col], X, missing="drop").fit()
        betas.loc[col] = res.params[factor_cols]
        idio_var[col] = res.resid.var(ddof=0)

    return betas, fac.cov(ddof=0), idio_var


def factor_model_cov(betas, factor_cov, idio_var):
    """Build covariance matrix implied by a factor model."""
    B = betas.values
    F = factor_cov.values
    D = np.diag(idio_var.values)
    cov = B @ F @ B.T + D
    return pd.DataFrame(cov, index=betas.index, columns=betas.index)


def factor_correlation(factors, factor_cols=None):
    """Compute correlation matrix across factors only."""
    if factor_cols is None:
        factor_cols = [c for c in factors.columns if c.lower() not in ("rf",)]
    fac = factors[factor_cols].copy()
    fac = _maybe_scale_factors(fac)
    return fac.corr()


def optimize_min_variance(cov):
    """Unconstrained minimum-variance portfolio (sum weights = 1)."""
    cov = np.asarray(cov)
    n = cov.shape[0]
    inv = np.linalg.pinv(cov)
    ones = np.ones(n)
    w = inv @ ones
    w = w / (ones @ inv @ ones)
    return w


def optimize_target_return(mu, cov, target):
    """Unconstrained mean-variance portfolio with target return."""
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = cov.shape[0]
    inv = np.linalg.pinv(cov)
    ones = np.ones(n)
    A = ones @ inv @ ones
    B = ones @ inv @ mu
    C = mu @ inv @ mu
    denom = A * C - B * B
    if denom == 0:
        return optimize_min_variance(cov)
    lam = (C - B * target) / denom
    gamma = (A * target - B) / denom
    w = inv @ (lam * ones + gamma * mu)
    return w


def optimize_max_sharpe(mu, cov, rf=0.0):
    """Unconstrained max-Sharpe portfolio."""
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    excess = mu - rf
    inv = np.linalg.pinv(cov)
    w = inv @ excess
    if w.sum() != 0:
        w = w / w.sum()
    return w


def optimize_long_only(mu, cov, target=None, rf=0.0, objective="min_var", n_random=2000, random_state=42):
    """Long-only optimization with optional target return.

    Uses SciPy SLSQP if available; otherwise falls back to random search.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = len(mu)

    if minimize is None:
        rng = np.random.default_rng(random_state)
        best_w = None
        best_val = np.inf
        for _ in range(n_random):
            w = rng.dirichlet(np.ones(n))
            if target is not None and (w @ mu) < target:
                continue
            if objective == "max_sharpe":
                vol = np.sqrt(w @ cov @ w)
                val = -((w @ mu) - rf) / vol if vol > 0 else np.inf
            else:
                val = w @ cov @ w
            if val < best_val:
                best_val = val
                best_w = w
        return best_w if best_w is not None else np.ones(n) / n

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target is not None:
        cons.append({"type": "ineq", "fun": lambda w: (w @ mu) - target})

    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    def obj_min_var(w):
        return w @ cov @ w

    def obj_max_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -((w @ mu) - rf) / vol if vol > 0 else np.inf

    obj = obj_max_sharpe if objective == "max_sharpe" else obj_min_var
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        return x0
    return res.x
