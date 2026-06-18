"""

run_parser.py — parses all downloaded transcripts.

Usage: python pipeline/run_parser.py

Run after run_scraper.py has populated data/raw/

"""

import logging

from src.parser import parse_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    logger.info("=== Starting transcript parsing ===")
    saved = parse_all(overwrite=False)
    logger.info(f"=== Done: {len(saved)} JSON files in data/transcripts/ ===")


if __name__ == "__main__":
    main()
