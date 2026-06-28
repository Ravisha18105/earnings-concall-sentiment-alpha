"""
run_backtest.py — runs the full factor backtest.
Usage: python pipeline/run_backtest.py
Run after run_signals.py has produced data/processed/signals.parquet
"""
import logging

from src.backtest import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Running factor backtest ===")

    for horizon in ["fwd_1d", "fwd_5d", "fwd_20d"]:
        logger.info(f"\n{'='*40}")
        logger.info(f"Horizon: {horizon}")
        logger.info(f"{'='*40}")
        results = run_backtest(return_col=horizon)
        logger.info(f"\nSummary ({horizon}):")
        logger.info(f"  Mean IC:     {results['mean_ic']}")
        logger.info(f"  ICIR:        {results['icir']}")
        logger.info(f"  t-stat:      {results['t_stat']}")
        logger.info(f"  Q5-Q1:       {results['q5_q1_spread']}")
        logger.info(f"  L/S Sharpe:  {results['ls_sharpe']}")

    logger.info("\n=== Charts saved to reports/ ===")
    logger.info("Files: quintile_returns_fwd_5d.png, ic_decay.png, ic_series.png")


if __name__ == "__main__":
    main()
