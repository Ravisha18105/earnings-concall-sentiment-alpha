"""
run_scorer.py — runs FinBERT scoring on all parsed transcripts.
Usage: python pipeline/run_scorer.py
Run after run_parser.py has populated data/transcripts/
"""
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.sentiment import score_all  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Starting FinBERT scoring ===")
    logger.info("This will take 45-90 min on CPU," " 8-15 min on MPS/GPU")
    df = score_all(overwrite=False)
    logger.info(f"=== Done: {len(df)} rows in scores.parquet ===")
    sample_cols = [
        "ticker",
        "quarter",
        "prep_net",
        "guidance_net",
        "lm_uncertainty",
    ]
    logger.info(
        "\nSample output:\n%s",
        df[sample_cols].head(10).to_string(),
    )


if __name__ == "__main__":
    main()
