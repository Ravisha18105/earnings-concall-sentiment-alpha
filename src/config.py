"""
config.py — single source of truth for all constants, paths, and settings.
Every other module imports from here. Never hardcode paths elsewhere.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_TRANSCRIPTS = ROOT / "data" / "transcripts"
DATA_PRICES = ROOT / "data" / "prices"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

for _dir in [DATA_RAW, DATA_TRANSCRIPTS, DATA_PRICES, DATA_PROCESSED, REPORTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "V",
    "UNH",
]

FORWARD_RETURN_DAYS = [1, 5, 20]

SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Consumer Discretionary",
    "NVDA": "Technology",
    "META": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "JPM": "Financials",
    "V": "Financials",
    "UNH": "Healthcare",
}

SIGNAL_WEIGHTS = {
    "guidance_net": 0.40,
    "qa_ceo_net": 0.30,
    "prep_net": 0.20,
    "lm_uncertainty": -0.10,
}

LOOKBACK_QUARTERS = 8
FORWARD_RETURN_DAYS = [1, 5, 20]
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_TOKENS = 512
FINBERT_BATCH_SIZE = 16

SIGNAL_WEIGHTS = {
    "finbert_qa_net": 0.40,
    "finbert_prep_net": 0.35,
    "lm_uncertainty": -0.25,
}

SCRAPE_DELAY_SEC = 1.5
SCRAPE_MAX_RETRY = 3

SEC_EDGAR_UA = os.getenv("SEC_EDGAR_UA", "your-email@example.com")

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "pipeline.log"),
    ],
)
