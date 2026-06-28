"""
signals.py — constructs the composite alpha signal from FinBERT scores.

Pipeline:
  1. Load scores.parquet
  2. Download / load price data
  3. Compute forward returns aligned to call dates
  4. Winsorize each sub-signal
  5. Cross-sectional z-score within each quarter
  6. Build weighted composite signal
  7. Re-z-score composite
  8. Assign quintile ranks
  9. Save signals.parquet
"""
import logging

import pandas as pd
from scipy import stats

from src.config import DATA_PROCESSED, SECTORS, SIGNAL_WEIGHTS
from src.prices import compute_forward_returns, load_prices

logger = logging.getLogger(__name__)

SIGNALS_PATH = DATA_PROCESSED / "signals.parquet"

WINSOR_LOW = 0.025  # 2.5th percentile
WINSOR_HIGH = 0.975  # 97.5th percentile


# ── step 1: winsorize ─────────────────────────────────────────────────────────


def winsorize_column(series: pd.Series) -> pd.Series:
    """Clip values to [2.5th, 97.5th] percentile to remove outliers."""
    lo = series.quantile(WINSOR_LOW)
    hi = series.quantile(WINSOR_HIGH)
    return series.clip(lo, hi)


# ── step 2: cross-sectional z-score ──────────────────────────────────────────


def zscore_cross_sectional(
    df: pd.DataFrame,
    col: str,
    group_col: str = "quarter",
) -> pd.Series:
    """
    Z-score a column cross-sectionally within each group (quarter).
    Handles the std=0 edge case (all stocks have same score → z=0).

    Returns a Series with the same index as df.
    """

    def _zscore(x):
        std = x.std()

        if pd.isna(std) or std < 1e-8:
            return pd.Series(0.0, index=x.index)

        return (x - x.mean()) / std

    return df.groupby(group_col)[col].transform(_zscore)


# ── step 3: composite signal ──────────────────────────────────────────────────


def build_composite(df: pd.DataFrame) -> pd.Series:
    """
    Weighted sum of z-scored sub-signals.
    Weights from config.SIGNAL_WEIGHTS.
    lm_uncertainty is SUBTRACTED (higher uncertainty → worse signal).
    """
    composite = pd.Series(0.0, index=df.index)

    for col, weight in SIGNAL_WEIGHTS.items():
        z_col = f"{col}_z"
        if z_col not in df.columns:
            logger.warning(f"Missing z-scored column: {z_col} — skipping")
            continue
        composite += weight * df[z_col]

    return composite


# ── step 4: quintile ranking ──────────────────────────────────────────────────


def assign_quintiles(series: pd.Series, n: int = 5) -> pd.Series:
    """Assign quintile ranks 1..n. Use n=5 (quintiles) for standard analysis."""
    try:
        return pd.qcut(series, q=n, labels=list(range(1, n + 1)), duplicates="drop").astype(float)
    except ValueError:
        return pd.cut(series.rank(method="first"), bins=n, labels=list(range(1, n + 1))).astype(
            float
        )


# ── step 5: preview IC ────────────────────────────────────────────────────────


def compute_ic(signal: pd.Series, fwd_return: pd.Series) -> dict:
    """
    Compute Spearman rank IC between signal and forward return.
    Returns dict with ic and p_value.
    """
    mask = signal.notna() & fwd_return.notna()
    if mask.sum() < 5:
        return {"ic": None, "p_value": None, "n": mask.sum()}
    ic, p = stats.spearmanr(signal[mask], fwd_return[mask])
    return {"ic": round(float(ic), 4), "p_value": round(float(p), 4), "n": int(mask.sum())}


# ── main pipeline ─────────────────────────────────────────────────────────────


def build_signals(overwrite: bool = False) -> pd.DataFrame:
    """
    Full signal construction pipeline.
    Returns signals DataFrame with all sub-signals, z-scores,
    composite score, quintile rank, and forward returns.
    """
    if SIGNALS_PATH.exists() and not overwrite:
        logger.info(f"Loading existing signals from {SIGNALS_PATH}")
        return pd.read_parquet(SIGNALS_PATH)

    # 1. Load scores
    scores_path = DATA_PROCESSED / "scores.parquet"
    df = pd.read_parquet(scores_path)
    logger.info(f"Loaded {len(df)} rows from scores.parquet")

    # 2. Add sector labels
    df["sector"] = df["ticker"].map(SECTORS).fillna("Unknown")

    # 3. Forward returns
    logger.info("Downloading / loading price data...")
    prices = load_prices()
    logger.info(f"Price shape: {prices.shape}")
    logger.info(f"Price columns: {list(prices.columns)[:20]}")
    logger.info(f"\n{prices.head()}")
    call_dates = df[["ticker", "date"]].copy()
    fwd_df = compute_forward_returns(prices, call_dates)

    # Merge forward returns onto scores
    df = df.merge(
        fwd_df[["ticker", "date", "fwd_1d", "fwd_5d", "fwd_20d"]],
        on=["ticker", "date"],
        how="left",
    )
    missing_fwd = df["fwd_1d"].isna().sum()
    if missing_fwd > 0:
        logger.warning(f"{missing_fwd} rows missing forward returns — check price coverage")

    # 4. Winsorize each raw sub-signal
    sub_signals = ["guidance_net", "qa_ceo_net", "prep_net", "lm_uncertainty"]
    for col in sub_signals:
        if col in df.columns:
            df[f"{col}_w"] = winsorize_column(df[col])
            logger.info(
                f"Winsorized {col}: [{df[col].min():.4f}, {df[col].max():.4f}] "
                f"→ [{df[f'{col}_w'].min():.4f}, {df[f'{col}_w'].max():.4f}]"
            )

    # 5. Cross-sectional z-score within each quarter
    for col in sub_signals:
        w_col = f"{col}_w"
        if w_col in df.columns:
            df[f"{col}_z"] = zscore_cross_sectional(df, w_col, group_col="quarter")

    # 6. Build composite signal
    df["raw_signal"] = build_composite(df)

    # 7. Re-z-score composite (overall, not per-quarter)
    # Per-quarter re-z-scoring with only 10 stocks gives unstable std
    sig_std = df["raw_signal"].std()
    sig_mean = df["raw_signal"].mean()
    df["composite_z"] = (df["raw_signal"] - sig_mean) / (sig_std + 1e-8)

    logger.info(f"raw_signal NaNs: {df['raw_signal'].isna().sum()} / {len(df)}")
    logger.info(f"composite_z NaNs: {df['composite_z'].isna().sum()} / {len(df)}")
    logger.info(f"\nComposite stats:\n{df['composite_z'].describe()}")

    # 8. Quintile ranks (cross-sectional within each quarter)
    df["quintile"] = df.groupby("quarter")["composite_z"].transform(
        lambda x: assign_quintiles(x, n=5)
    )

    # 9. Preview IC
    for days in [1, 5, 20]:
        fwd_col = f"fwd_{days}d"
        if fwd_col in df.columns:
            ic_result = compute_ic(df["composite_z"], df[fwd_col])
            logger.info(
                f"IC (fwd_{days}d): {ic_result['ic']} "
                f"(p={ic_result['p_value']}, n={ic_result['n']})"
            )

    # 10. Save
    df.to_parquet(SIGNALS_PATH, index=False)
    logger.info(f"Saved {len(df)} rows to {SIGNALS_PATH}")

    # Print summary
    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    """Print a clean summary table of the signal DataFrame."""
    display_cols = [
        "ticker",
        "quarter",
        "guidance_net",
        "qa_ceo_net",
        "composite_z",
        "quintile",
        "fwd_1d",
        "fwd_5d",
    ]
    available = [c for c in display_cols if c in df.columns]
    logger.info(f"\nSignal summary (first 15 rows):\n{df[available].head(15).to_string()}")

    logger.info(f"\nComposite z-score stats:\n{df['composite_z'].describe().round(4)}")

    if "fwd_5d" in df.columns:
        logger.info("\nMean fwd_5d return by quintile:")
        q_ret = df.groupby("quintile")["fwd_5d"].mean().round(4)
        logger.info(q_ret.to_string())
