"""
edgar_utils.py — utility functions for navigating SEC EDGAR programmatically.
"""
import logging

import requests

from src.config import SEC_EDGAR_UA

logger = logging.getLogger(__name__)
HEADERS = {"User-Agent": SEC_EDGAR_UA}


def get_cik(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"

    r = requests.get(url, headers=HEADERS, timeout=15)

    r.raise_for_status()

    data = r.json()

    ticker = ticker.upper()

    for company in data.values():
        if company["ticker"] == ticker:
            return str(company["cik_str"]).zfill(10)

    raise ValueError(f"Ticker not found: {ticker}")


def get_submissions(cik: str) -> dict:
    """Fetch all filings metadata for a CIK from EDGAR submissions API."""
    logger.info("Fetching submissions for CIK %s", cik)
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def filter_8k_filings(submissions: dict) -> list[dict]:
    recent = submissions["filings"]["recent"]

    forms = recent["form"]
    dates = recent["filingDate"]
    accessions = recent["accessionNumber"]
    primary_docs = recent["primaryDocument"]

    results = []

    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form == "8-K":
            results.append(
                {
                    "form": form,
                    "date": date,
                    "accession": acc,
                    "primary_document": doc,
                }
            )

    return results


def get_filing_index_url(cik: str, accession: str) -> str:
    clean_acc = accession.replace("-", "")

    return (
        f"https://www.sec.gov/Archives/"
        f"edgar/data/{int(cik)}/"
        f"{clean_acc}/"
        f"{clean_acc}-index.htm"
    )


def build_primary_document_url(
    cik: str,
    accession: str,
    primary_document: str,
) -> str:
    clean_acc = accession.replace("-", "")

    return (
        f"https://www.sec.gov/Archives/"
        f"edgar/data/{int(cik)}/"
        f"{clean_acc}/"
        f"{primary_document}"
    )
