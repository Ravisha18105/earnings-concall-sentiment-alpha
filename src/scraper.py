# src/scraper.py — rewritten for Motley Fool

import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import DATA_RAW

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
}

# Manually curated Motley Fool transcript URLs for your tickers
# Run build_url_list() once to generate these, then hardcode
TRANSCRIPT_URLS = {
    "AAPL": [
        "https://www.fool.com/earnings/call-transcripts/2026/04/30/apple-aapl-q2-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2026/01/29/apple-aapl-q1-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/10/31/apple-q4-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/08/01/apple-aapl-q3-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/10/31/apple-aapl-q4-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2023/02/02/apple-aapl-q1-2023-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2022/10/27/apple-aapl-q4-2022-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2022/01/28/apple-aapl-q1-2022-earnings-call-transcript/",  # noqa: E501
    ],
    "MSFT": [
        "https://www.fool.com/earnings/call-transcripts/2026/04/29/microsoft-msft-q3-2026-earnings-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2026/01/28/microsoft-msft-q2-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/10/29/microsoft-msft-q1-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/08/05/microsoft-msft-q4-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/01/29/microsoft-msft-q2-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/07/30/microsoft-msft-q4-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/04/25/microsoft-msft-q3-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2023/07/25/microsoft-msft-q4-2023-earnings-call-transcript/",  # noqa: E501
    ],
    "GOOGL": [
        "https://www.fool.com/earnings/call-transcripts/2026/04/29/alphabet-googl-q1-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2026/02/04/alphabet-googl-q4-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/11/27/alphabet-googl-q3-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/07/23/alphabet-googl-q2-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/07/23/alphabet-googl-q2-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/04/25/alphabet-googl-q1-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/01/30/alphabet-googl-q4-2023-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2023/07/25/alphabet-googl-q2-2023-earnings-call-transcript/",  # noqa: E501
    ],
    "AMZN": [
        "https://www.fool.com/earnings/call-transcripts/2026/04/29/amazon-amzn-q1-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2026/02/05/amazon-amzn-q4-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/10/31/amazon-amzn-q3-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/02/06/amazoncom-amzn-q4-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/08/01/amazoncom-amzn-q2-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/04/30/amazoncom-amzn-q1-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/02/01/amazoncom-amzn-q4-2023-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2023/10/26/amazoncom-amzn-q3-2023-earnings-call-transcript/",  # noqa: E501
    ],
    "NVDA": [
        "https://www.fool.com/earnings/call-transcripts/2026/02/25/nvidia-nvda-q4-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/11/19/nvidia-nvda-q3-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/02/26/nvidia-nvda-q4-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/08/28/nvidia-nvda-q2-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2023/11/22/nvidia-nvda-q3-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2023/08/23/nvidia-nvda-q2-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2023/05/24/nvidia-nvda-q1-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2023/02/22/nvidia-nvda-q4-2023-earnings-call-transcript/",  # noqa: E501
    ],
    "META": [
        "https://www.fool.com/earnings/call-transcripts/2026/04/29/meta-meta-q1-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2026/01/28/meta-meta-q4-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/01/29/meta-platforms-meta-q4-2024-earnings-call-transcri/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/10/30/meta-platforms-meta-q3-2024-earnings-call-transcri/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/10/29/meta-platforms-meta-q3-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2022/07/27/meta-platforms-inc-meta-q2-2022-earnings-call-tran/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2022/04/27/meta-platforms-inc-fb-q1-2022-earnings-call-transc/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2022/02/03/facebook-fb-q4-2021-earnings-call-transcript/",  # noqa: E501
    ],
    "TSLA": [
        "https://www.fool.com/earnings/call-transcripts/2026/01/28/tesla-tsla-q4-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/10/22/tesla-tsla-q3-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/07/23/tesla-tsla-q2-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/01/29/tesla-tsla-q4-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2026/04/22/tesla-tsla-q3-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/07/24/tesla-tsla-q2-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/04/23/tesla-tsla-q1-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/01/24/tesla-tsla-q4-2023-earnings-call-transcript/",  # noqa: E501
    ],
    "JPM": [
        "https://www.fool.com/earnings/call-transcripts/2026/04/21/jpmorgan-jpm-q1-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2026/01/15/j-p-morgan-chase-jpm-q4-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/08/04/jpmorgan-jpm-q2-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/01/15/jpmorgan-chase-jpm-q4-2024-earnings-call-transcrip/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/10/11/jpmorgan-chase-jpm-q3-2024-earnings-call-transcrip/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/07/12/jpmorgan-chase-jpm-q2-2024-earnings-call-transcrip/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/04/12/jpmorgan-chase-jpm-q1-2024-earnings-call-transcrip/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/01/12/jpmorgan-chase-jpm-q4-2023-earnings-call-transcrip/",  # noqa: E501
    ],
    "V": [
        "https://www.fool.com/earnings/call-transcripts/2026/04/29/visa-v-q2-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2026/01/30/visa-v-q1-2026-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/10/28/visa-v-q4-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/08/04/visa-v-q3-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/01/30/visa-v-q1-2025-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/10/29/visa-v-q4-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/07/23/visa-v-q3-2024-earnings-call-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/04/23/visa-v-q2-2024-earnings-call-transcript/",  # noqa: E501
    ],
    "UNH": [
        "https://www.fool.com/earnings/call-transcripts/2026/04/21/unitedhealth-unh-q1-2026-earnings-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/08/06/unitedhealth-unh-q2-2025-earnings-transcript/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2025/01/16/unitedhealth-group-unh-q4-2024-earnings-call-trans/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/10/15/unitedhealth-group-unh-q3-2024-earnings-call-trans/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/07/16/unitedhealth-group-unh-q2-2024-earnings-call-trans/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/04/16/unitedhealth-group-unh-q1-2024-earnings-call-trans/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2024/01/12/unitedhealth-group-unh-q4-2023-earnings-call-trans/",  # noqa: E501
        "https://www.fool.com/earnings/call-transcripts/2023/07/14/unitedhealth-group-unh-q2-2023-earnings-call-trans/",  # noqa: E501
    ],
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=20)
    if r.status_code == 429:
        time.sleep(30)
        r.raise_for_status()
    r.raise_for_status()
    return r


def fetch_motley_fool_transcript(ticker: str, url: str) -> Optional[Path]:
    """
    Fetch one Motley Fool transcript page and save raw HTML.
    Extracts date from URL for filename.
    """
    # Parse date from URL: .../2024/02/01/apple-... → 2024-02-01
    parts = url.split("/")
    try:
        date_str = f"{parts[6]}-{parts[7]}-{parts[8]}"
    except IndexError:
        date_str = "unknown"

    fname = f"{ticker}_{date_str}_motleyfool.html"
    save_path = DATA_RAW / fname
    meta_path = DATA_RAW / fname.replace(".html", "_meta.json")

    if save_path.exists():
        logger.info(f"Already downloaded: {fname} — skipping")
        return save_path

    try:
        r = _get(url)

        # Quick transcript check before saving
        text = BeautifulSoup(r.text, "lxml").get_text()
        markers = ["operator", "prepared remarks", "question", "q&a"]
        if not any(m in text.lower() for m in markers):
            logger.warning(f"⚠ {fname} — no transcript markers found")
            return None

        save_path.write_text(r.text, encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "ticker": ticker,
                    "date": date_str,
                    "source": "motley_fool",
                    "url": url,
                    "word_count": len(text.split()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        logger.info(f"Saved: {fname} ({len(text.split()):,} words)")
        time.sleep(random.uniform(2.0, 4.0))
        return save_path

    except Exception as e:
        logger.error(f"Failed: {ticker} {date_str}: {e}")
        raise


def scrape_ticker(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    **kwargs,
) -> list[Path]:
    """Scrape all transcript URLs for a ticker."""
    urls = TRANSCRIPT_URLS.get(ticker, [])
    if not urls:
        logger.warning(f"No URLs configured for {ticker}")
        return []

    saved = []
    for url in urls:
        try:
            path = fetch_motley_fool_transcript(ticker, url)
            if path:
                saved.append(path)
        except Exception as e:
            logger.error(f"Skipping {url}: {e}")

    logger.info(f"{ticker}: saved {len(saved)}/{len(urls)} transcripts")
    return saved
