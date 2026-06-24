import pandas as pd
import yfinance as yf

raw = yf.download(
    ["AAPL", "MSFT"], start="2024-01-01", end="2024-03-01", auto_adjust=True, progress=False
)
print("Columns type:", type(raw.columns))
print("Columns:", raw.columns.tolist()[:6])
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"]
    print("Close shape:", prices.shape)
    print(prices.head(3))
