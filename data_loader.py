import yfinance as yf
from datetime import datetime

def download_stock_data(ticker: str, start="2020-01-01", end=None):
    if end is None:
        end = datetime.utcnow().strftime("%Y-%m-%d")

    df = yf.Ticker(ticker).history(start=start, end=end)

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    df = df.reset_index()
    df.columns = [c.replace(" ", "_") for c in df.columns]

    return df