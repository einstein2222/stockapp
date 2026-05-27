from pathlib import Path
import time

import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "NFLX", "JPM", "XOM",
    "UNH", "COST"
]

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_and_save(ticker: str, start="2020-01-01"):
    path = DATA_DIR / f"{ticker.upper()}.csv"

    if path.exists():
        print(f"{ticker}: already cached")
        return

    df = yf.download(
        ticker,
        start=start,
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if df is None or df.empty:
        raise ValueError(f"Failed to download data for {ticker}")

    df = df.reset_index()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df.to_csv(path, index=False)
    print(f"{ticker}: saved to {path}")


if __name__ == "__main__":
    for t in TICKERS:
        try:
            fetch_and_save(t)
            time.sleep(1.2)
        except Exception as e:
            print(f"{t}: {e}")