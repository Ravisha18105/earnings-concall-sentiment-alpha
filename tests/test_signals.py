"""test_signals.py — unit tests for signal construction."""
import pandas as pd

from src.signals import assign_quintiles, build_composite, winsorize_column, zscore_cross_sectional


def make_df():
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "quarter": ["Q1-2024"] * 5,
            "guidance_net_w": [0.1, 0.3, 0.5, 0.7, 0.9],
            "qa_ceo_net_w": [0.2, 0.4, 0.3, 0.6, 0.8],
            "prep_net_w": [0.3, 0.3, 0.4, 0.5, 0.6],
            "lm_uncertainty_w": [0.02, 0.01, 0.03, 0.01, 0.02],
        }
    )


def test_winsorize_clips_outliers():
    s = pd.Series([0, 1, 2, 3, 100])
    w = winsorize_column(s)
    assert w.max() < 100


def test_zscore_mean_zero():
    df = make_df()
    z = zscore_cross_sectional(df, "guidance_net_w", "quarter")
    assert abs(z.mean()) < 1e-8


def test_zscore_std_one():
    df = make_df()
    z = zscore_cross_sectional(df, "guidance_net_w", "quarter")
    assert abs(z.std() - 1.0) < 0.01


def test_composite_shape():
    df = make_df()
    for col in ["guidance_net", "qa_ceo_net", "prep_net", "lm_uncertainty"]:
        df[f"{col}_z"] = zscore_cross_sectional(df, f"{col}_w", "quarter")
    composite = build_composite(df)
    assert len(composite) == len(df)
    assert not composite.isna().any()


def test_quintiles_range():
    s = pd.Series([0.1, 0.2, 0.5, 0.8, 0.9])
    q = assign_quintiles(s)
    assert set(q.dropna().astype(int)) == {1, 2, 3, 4, 5}
