"""
run_scraper.py — runs the Motley Fool transcript scraper
Usage: python pipeline/run_scraper.py
"""
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import TICKERS  # noqa: E402
from src.scraper import scrape_ticker  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    logger.info(f"Starting Motley Fool scrape: {len(TICKERS)} tickers")

    total_saved = 0

    for ticker in TICKERS:
        logger.info(f"=== Scraping {ticker} ===")

        try:
            paths = scrape_ticker(ticker)
            total_saved += len(paths)

            logger.info(f"{ticker}: saved {len(paths)} transcripts")

        except Exception as e:
            logger.error(f"{ticker} failed entirely: {e}")

    logger.info(f"DONE — total files saved: {total_saved}")


if __name__ == "__main__":
    main()
