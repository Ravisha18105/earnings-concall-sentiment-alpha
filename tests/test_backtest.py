"""test_backtest.py — unit tests for backtest functions."""
import numpy as np
import pandas as pd

from src.backtest import (
    compute_ic_series,
    compute_icir,
    compute_ls_portfolio,
    compute_quintile_returns,
)


def make_df():
    return pd.DataFrame(
        {
            "quarter": ["Q1-2024"] * 5 + ["Q2-2024"] * 5,
            "ticker": ["A", "B", "C", "D", "E"] * 2,
            "composite_z": [2.0, 1.0, 0.0, -1.0, -2.0] * 2,
            "quintile": [5, 4, 3, 2, 1] * 2,
            "fwd_5d": [0.08, 0.04, 0.01, -0.02, -0.06, 0.06, 0.03, 0.00, -0.01, -0.05],
        }
    )


def test_ic_series_positive():
    df = make_df()
    ic = compute_ic_series(df, "composite_z", "fwd_5d")
    assert ic.mean() > 0, "Signal should have positive IC with this synthetic data"


def test_icir_positive():
    df = make_df()
    ic = compute_ic_series(df, "composite_z", "fwd_5d")
    result = compute_icir(ic)
    assert result["icir"] > 0


def test_quintile_returns_monotonic():
    df = make_df()
    qret = compute_quintile_returns(df, "fwd_5d")
    returns = qret.sort_values("quintile")["mean_return"].values
    assert returns[-1] > returns[0], "Q5 should have higher return than Q1"


def test_ls_portfolio_positive_sharpe():
    df = make_df()
    ls = compute_ls_portfolio(df, "fwd_5d")
    assert ls["sharpe"] > 0, "L/S Sharpe should be positive with this synthetic data"


def test_ic_handles_nans():
    df = make_df()
    df.loc[0, "fwd_5d"] = np.nan
    ic = compute_ic_series(df, "composite_z", "fwd_5d")
    assert not ic.isna().all(), "IC should handle NaN forward returns"
