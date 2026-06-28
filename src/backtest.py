"""
backtest.py — factor backtest for the earnings sentiment alpha signal.

Implements:
  - IC / ICIR analysis (Spearman rank correlation)
  - Quintile return analysis (Q1–Q5 mean returns)
  - Factor decay curve (IC at t+1, t+5, t+10, t+20)
  - Long-short portfolio Sharpe ratio
  - Alphalens full tearsheet (if alphalens-reloaded is available)
  - Chart generation saved to reports/

Designed to work even without Alphalens installed.
"""
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA_PROCESSED, REPORTS_DIR

matplotlib.use("Agg")  # non-interactive backend for saving to file

logger = logging.getLogger(__name__)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── IC analysis ───────────────────────────────────────────────────────────────


def compute_ic_series(df: pd.DataFrame, signal_col: str, return_col: str) -> pd.Series:
    """
    Compute per-quarter Spearman IC between signal and forward return.
    Returns a Series indexed by quarter.
    """

    def _ic(group):
        valid = group[[signal_col, return_col]].dropna()
        if len(valid) < 4:
            return np.nan
        if valid[signal_col].nunique() < 2 or valid[return_col].nunique() < 2:
            return np.nan
        ic, _ = stats.spearmanr(
            valid[signal_col],
            valid[return_col],
        )
        return float(ic)

    return (
        df[["quarter", signal_col, return_col]]
        .groupby("quarter")
        .apply(_ic, include_groups=False)
        .rename("IC")
    )


def compute_icir(ic_series: pd.Series) -> dict:
    """ICIR = mean(IC) / std(IC). Returns dict with ic, icir, t_stat."""
    clean = ic_series.dropna()
    if len(clean) == 0:
        return {"mean_ic": None, "std_ic": None, "icir": None, "t_stat": None, "n_quarters": 0}
    mean_ic = float(clean.mean())
    std_ic = float(clean.std())
    icir = mean_ic / (std_ic + 1e-8)
    t_stat = mean_ic / (std_ic / np.sqrt(len(clean)) + 1e-8)
    return {
        "mean_ic": round(mean_ic, 4),
        "std_ic": round(std_ic, 4),
        "icir": round(icir, 4),
        "t_stat": round(t_stat, 4),
        "n_quarters": len(clean),
    }


# ── quintile returns ──────────────────────────────────────────────────────────


def compute_quintile_returns(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    """
    Mean forward return by quintile.
    Returns DataFrame with columns [quintile, mean_return, count, std_return].
    """
    valid = df.dropna(subset=["quintile", return_col])
    grouped = (
        valid.groupby("quintile")[return_col]
        .agg(
            mean_return="mean",
            std_return="std",
            count="count",
        )
        .reset_index()
    )
    grouped["quintile"] = grouped["quintile"].astype(int)
    return grouped.sort_values("quintile")


# ── factor decay ──────────────────────────────────────────────────────────────


def compute_factor_decay(df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    """
    Compute IC at multiple forward horizons to show factor decay.
    Returns DataFrame with columns [horizon_days, mean_ic, icir].
    """
    horizons = [1, 5, 10, 20]
    results = []

    for days in horizons:
        col = f"fwd_{days}d"
        if col not in df.columns:
            continue
        ic_series = compute_ic_series(df, signal_col, col)
        stats_ = compute_icir(ic_series)
        results.append(
            {
                "horizon_days": days,
                "mean_ic": stats_["mean_ic"],
                "icir": stats_["icir"],
                "n_quarters": stats_["n_quarters"],
            }
        )

    return pd.DataFrame(results)


# ── long-short portfolio ──────────────────────────────────────────────────────


def compute_ls_portfolio(df, return_col="fwd_5d"):
    valid = df.dropna(subset=["quintile", return_col])
    ls_returns = []
    for quarter, group in valid.groupby("quarter"):
        max_q = group["quintile"].max()
        min_q = group["quintile"].min()
        long_ret = group[group["quintile"] == max_q][return_col].mean()
        short_ret = group[group["quintile"] == min_q][return_col].mean()
        if pd.notna(long_ret) and pd.notna(short_ret):
            ls_returns.append(long_ret - short_ret)
    returns = np.array(ls_returns)
    mean_r = float(returns.mean())
    std_r = float(returns.std())

    # Annualize: earnings calls are ~quarterly so 4 periods/year
    ann_mean = mean_r * 4
    ann_std = std_r * np.sqrt(4)
    sharpe = ann_mean / (ann_std + 1e-8)

    return {
        "sharpe": round(sharpe, 4),
        "mean_return": round(mean_r, 4),
        "std_return": round(std_r, 4),
        "ann_return": round(ann_mean, 4),
        "n_periods": len(ls_returns),
    }


# ── charts ────────────────────────────────────────────────────────────────────


def plot_quintile_returns(qret: pd.DataFrame, return_col: str, save_path: Path):
    """Bar chart: mean return per quintile. Q1=red, Q5=green."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#E24B4A", "#F0997B", "#888780", "#9FE1CB", "#1D9E75"]
    bars = ax.bar(
        qret["quintile"].astype(str),
        qret["mean_return"] * 100,  # convert to %
        color=colors[: len(qret)],
        width=0.6,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, val in zip(bars, qret["mean_return"]):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + 0.05,
            f"{val*100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.axhline(0, color="#444441", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Signal quintile (1=bearish, 5=bullish)", fontsize=11)
    ax.set_ylabel(f"Mean {return_col} return (%)", fontsize=11)
    ax.set_title(
        f"Mean forward return by quintile ({return_col})", fontsize=12, fontweight="normal"
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_ic_decay(decay_df: pd.DataFrame, save_path: Path):
    """Line chart: IC at different horizons (factor decay curve)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        decay_df["horizon_days"],
        decay_df["mean_ic"],
        marker="o",
        color="#1D9E75",
        linewidth=2,
        markersize=6,
    )
    ax.axhline(0, color="#444441", linewidth=0.8, linestyle="--")
    for _, row in decay_df.iterrows():
        if pd.notna(row["mean_ic"]):
            ax.annotate(
                f"{row['mean_ic']:.3f}",
                (row["horizon_days"], row["mean_ic"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=10,
            )
    ax.set_xlabel("Forward horizon (trading days)", fontsize=11)
    ax.set_ylabel("Mean IC (Spearman)", fontsize=11)
    ax.set_title("Factor decay — IC vs forward horizon", fontsize=12, fontweight="normal")
    ax.set_xticks(decay_df["horizon_days"].tolist())
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def plot_ic_series(ic_series: pd.Series, save_path: Path):
    """Bar chart of per-quarter IC values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    clean = ic_series.dropna()
    colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in clean.values]
    ax.bar(range(len(clean)), clean.values, color=colors, width=0.6, edgecolor="white")
    ax.axhline(0, color="#444441", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(clean)))
    ax.set_xticklabels(clean.index.tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("IC (Spearman)", fontsize=11)
    ax.set_title("Per-quarter IC — earnings sentiment signal", fontsize=12, fontweight="normal")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


# ── alphalens tearsheet (optional) ───────────────────────────────────────────


def run_alphalens_tearsheet(df: pd.DataFrame, prices: pd.DataFrame) -> bool:
    """
    Attempt to run full Alphalens tearsheet.
    Returns True if successful, False if Alphalens unavailable or errors.
    """
    try:
        import alphalens
    except ImportError:
        logger.warning(
            "alphalens-reloaded not installed — skipping tearsheet. "
            "pip install alphalens-reloaded"
        )
        return False

    try:
        valid = df.dropna(subset=["composite_z", "date", "ticker"])

        # Format factor: Series with MultiIndex (date, asset)
        factor = valid.set_index(["date", "ticker"])["composite_z"]
        factor.index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(d), t) for d, t in factor.index],
            names=["date", "asset"],
        )
        factor = factor.sort_index()

        # Prices: DatetimeIndex × ticker columns
        prices_fmt = prices.copy()
        prices_fmt.index = pd.to_datetime(prices_fmt.index)

        factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
            factor=factor,
            prices=prices_fmt,
            quantiles=5,
            periods=(1, 5, 20),
            max_loss=0.5,  # allow up to 50% data loss (small dataset)
        )

        ic = alphalens.performance.factor_information_coefficient(factor_data)
        logger.info(f"Alphalens mean IC:\n{ic.mean().round(4)}")

        fig = alphalens.plotting.create_full_tear_sheet(
            factor_data, long_short=True, group_neutral=False
        )
        out = REPORTS_DIR / "alphalens_tearsheet.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Alphalens tearsheet saved: {out}")
        return True

    except Exception as e:
        logger.warning(f"Alphalens tearsheet failed: {e}")
        logger.info("Falling back to manual backtest charts")
        return False


# ── main ──────────────────────────────────────────────────────────────────────


def run_backtest(return_col: str = "fwd_5d") -> dict:
    """
    Full backtest pipeline. Returns dict of all key metrics.
    """
    signals_path = DATA_PROCESSED / "signals.parquet"
    df = pd.read_parquet(signals_path)
    logger.info(f"Loaded {len(df)} rows from signals.parquet")

    valid = df.dropna(subset=["composite_z"])
    valid = valid[valid["composite_z"].abs() > 1e-10].reset_index(drop=True)
    logger.info(f"Valid rows after dropping zero z-scores: {len(valid)}")

    # Recompute tertiles per-quarter on the clean data
    def assign_tertile(x):
        try:
            return pd.qcut(x, q=3, labels=[1, 2, 3], duplicates="drop").astype(float)
        except ValueError:
            return pd.cut(x.rank(method="first"), bins=3, labels=[1, 2, 3]).astype(float)

    valid["quintile"] = valid.groupby("quarter")["composite_z"].transform(assign_tertile)
    distribution = valid.groupby(["quarter", "quintile"]).size().unstack(fill_value=0)
    logger.info("Tertile distribution:\n%s", distribution)

    # 1. IC series + ICIR
    ic_series = compute_ic_series(valid, "composite_z", return_col)
    icir_stats = compute_icir(ic_series)
    logger.info(f"\nIC analysis ({return_col}):")
    for k, v in icir_stats.items():
        logger.info(f"  {k}: {v}")

    # 2. Quintile returns
    qret = compute_quintile_returns(valid, return_col)
    logger.info(f"\nQuintile returns ({return_col}):\n{qret.to_string(index=False)}")
    if len(qret) >= 2:
        spread = (
            qret[qret.quintile == qret.quintile.max()]["mean_return"].values[0]
            - qret[qret.quintile == qret.quintile.min()]["mean_return"].values[0]
        )
    else:
        spread = None
    if spread is not None:
        logger.info(f"  Q5 - Q1 spread: {spread*100:.3f}%")

    # 3. Factor decay
    decay = compute_factor_decay(valid, "composite_z")
    logger.info(f"\nFactor decay:\n{decay.to_string(index=False)}")

    # 4. Long-short Sharpe
    ls = compute_ls_portfolio(valid, return_col)
    logger.info(f"\nLong-short portfolio: {ls}")

    # 5. Save charts
    plot_quintile_returns(qret, return_col, REPORTS_DIR / f"quintile_returns_{return_col}.png")
    plot_ic_decay(decay, REPORTS_DIR / "ic_decay.png")
    plot_ic_series(ic_series, REPORTS_DIR / "ic_series.png")

    # 6. Try Alphalens tearsheet
    try:
        prices = pd.read_parquet(DATA_PROCESSED / "prices.parquet")
        run_alphalens_tearsheet(df, prices)
    except Exception as e:
        logger.warning(f"Could not run Alphalens: {e}")

    results = {
        "mean_ic": icir_stats["mean_ic"],
        "icir": icir_stats["icir"],
        "t_stat": icir_stats["t_stat"],
        "n_quarters": icir_stats["n_quarters"],
        "ls_sharpe": ls["sharpe"],
        "ls_ann_return": ls["ann_return"],
        "q5_q1_spread": round(spread, 4) if spread else None,
        "return_col": return_col,
    }
    return results
