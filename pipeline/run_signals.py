"""
run_signals.py — builds the composite alpha signal.
Usage: python pipeline/run_signals.py
Run after run_scorer.py has produced data/processed/scores.parquet
"""
import logging

from src.signals import build_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    logger.info("=== Building alpha signal ===")

    df = build_signals(overwrite=True)

    logger.info(f"=== Done: {len(df)} rows in signals.parquet ===")


if __name__ == "__main__":
    main()
