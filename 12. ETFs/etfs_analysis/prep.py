"""Data preparation utilities for ETF returns and universe selection."""

import pandas as pd


def select_top_etfs_by_category(df, top_n=5, score_cols=("AUM", "ADV")):
    """Select top-N ETFs per category using a simple size/liquidity score."""
    df = df.copy()
    score = pd.Series(0, index=df.index, dtype=float)
    for col in score_cols:
        if col in df.columns:
            score += pd.to_numeric(df[col], errors="coerce").fillna(0)
    if score.sum() == 0:
        df["_score"] = range(len(df))
    else:
        df["_score"] = score
    df = df.sort_values(["CATEGORY", "_score"], ascending=[True, False])
    out = df.groupby("CATEGORY", group_keys=False).head(top_n)
    return out.drop(columns=["_score"])


def build_returns_panel(df_etf, tickers, min_history=252, fill_method="none"):
    """Pivot a date x ticker return panel with de-duplication and imputation.

    fill_method options: "none", "mean", "ffill", "zero".
    """
    df = df_etf.copy()
    df = df[df["TICKER"].isin(tickers)]
    df = df.sort_values("date")
    df = df.groupby(["date", "TICKER"], as_index=False)["RET"].last()
    ret = df.pivot(index="date", columns="TICKER", values="RET")
    enough = ret.notna().sum() >= min_history
    ret = ret.loc[:, enough]
    if fill_method == "mean":
        ret = ret.fillna(ret.mean())
    elif fill_method == "ffill":
        ret = ret.fillna(method="ffill")
    elif fill_method == "zero":
        ret = ret.fillna(0)
    return ret.sort_index()
