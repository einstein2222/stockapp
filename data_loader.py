from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def download_stock_data(ticker: str, start="2020-01-01", allow_download=True):
    ticker = ticker.upper().strip()
    path = DATA_DIR / f"{ticker}.csv"

    if path.exists():
        df = pd.read_csv(path)
    elif allow_download:
        df = yf.download(
            ticker,
            start=start,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df is None or df.empty:
            raise ValueError(f"No data returned for {ticker}")

        df = df.reset_index()
        df.columns = [c.replace(" ", "_") for c in df.columns]
        df.to_csv(path, index=False)
    else:
        raise FileNotFoundError(f"Missing cached CSV for {ticker}: {path}")

    if "Date" not in df.columns:
        raise ValueError(f"{path} must contain a Date column")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] >= pd.to_datetime(start)].reset_index(drop=True)

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker} missing columns: {missing}")

    return df