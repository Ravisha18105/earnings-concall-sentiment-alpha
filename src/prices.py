"""
prices.py — downloads adjusted close prices from yfinance and
computes forward returns aligned to earnings call dates.

Forward returns are calculated from t+1 close to avoid lookahead bias.
The call date itself (t=0) already has a price reaction baked in.
"""
import logging

import pandas as pd
import yfinance as yf

from src.config import DATA_PROCESSED, FORWARD_RETURN_DAYS, TICKERS

logger = logging.getLogger(__name__)

PRICES_PATH = DATA_PROCESSED / "prices.parquet"


def download_prices(tickers=None, start="2021-10-01", end="2025-06-30"):
    tickers = tickers or TICKERS
    logger.info(f"Downloading prices for {len(tickers)} tickers...")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance MultiIndex handling — THIS is the bug
    # When multiple tickers: columns are MultiIndex (metric, ticker)
    # When single ticker: columns are flat ['Open','High','Low','Close','Volume']
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]  # shape: (dates, tickers)
    else:
        # Single ticker case — raw has flat columns
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    logger.info(f"Price shape: {prices.shape} | columns: {list(prices.columns)}")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.ffill()
    prices.to_parquet(PRICES_PATH)
    return prices


def compute_forward_returns(
    prices: pd.DataFrame,
    call_dates: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (ticker, call_date) pair, compute forward returns
    at 1, 5, and 20 trading days from t+1 (day after the call).

    call_dates: DataFrame with columns [ticker, date]
    Returns: call_dates with fwd_1d, fwd_5d, fwd_20d columns added.
    """
    results = []

    for _, row in call_dates.iterrows():
        ticker = row["ticker"]
        date = pd.Timestamp(row["date"])

        if ticker not in prices.columns:
            logger.warning(f"{ticker} not in price data")
            continue

        ticker_prices = prices[ticker].dropna()
        trading_days = ticker_prices.index

        # Find t+1 (first trading day after call date)
        future_days = trading_days[trading_days > date]
        if len(future_days) == 0:
            logger.warning(f"No future trading days for {ticker} after {date}")
            continue

        t1_date = future_days[0]
        t1_price = ticker_prices.loc[t1_date]

        result = {**row.to_dict()}

        for days in FORWARD_RETURN_DAYS:
            target_position = days
            if len(future_days) > target_position:
                target_date = future_days[target_position]
                target_price = ticker_prices.loc[target_date]
                forward_return = (target_price / t1_price) - 1
                result[f"fwd_{days}d"] = round(float(forward_return), 6)
            else:
                result[f"fwd_{days}d"] = None

        results.append(result)

    return pd.DataFrame(results)


def load_prices() -> pd.DataFrame:
    """Load prices from parquet if it exists, else download."""
    if PRICES_PATH.exists():
        return pd.read_parquet(PRICES_PATH)
    return download_prices()
