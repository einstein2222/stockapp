from pathlib import Path
import time

import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "NFLX", "JPM", "XOM",
    "UNH", "COST", "HD", "PG", "KO",
    "NKE", "IBM", "INTC", "ORCL", "AMD"
]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_one(ticker: str, start="2020-01-01", retries=3):
    path = DATA_DIR / f"{ticker}.csv"

    if path.exists():
        print(f"{ticker}: already cached")
        return True

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if df is None or df.empty:
                raise ValueError("empty dataframe")

            df = df.reset_index()
            df.columns = [c.replace(" ", "_") for c in df.columns]
            df.to_csv(path, index=False)
            print(f"{ticker}: saved")
            return True
        except Exception as e:
            print(f"{ticker}: attempt {attempt} failed -> {e}")
            time.sleep(1.5)

    print(f"{ticker}: skipped")
    return False


if __name__ == "__main__":
    ok = 0
    for t in TICKERS:
        if fetch_one(t):
            ok += 1

    print(f"\nSaved {ok}/{len(TICKERS)} tickers.")