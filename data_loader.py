from pathlib import Path
import pandas as pd

HF_DATASET_PATH = "hf://datasets/usamaahmedsh/elliott-wave-market-data-complete/market_data_1d.parquet"

def download_stock_data(ticker: str, start="2020-01-01", end=None):
    ticker = ticker.upper().strip()

    df = pd.read_parquet(HF_DATASET_PATH)

    # normalize column names from the dataset
    df.columns = [c.lower() for c in df.columns]

    # expected columns in the HF dataset:
    # datetime, open, high, low, close, volume, ticker, interval, source_category
    if "ticker" not in df.columns:
        raise ValueError("Dataset does not contain ticker column")

    df = df[df["ticker"].astype(str).str.upper() == ticker].copy()

    if df.empty:
        raise ValueError(f"No data found for {ticker} in the Hugging Face dataset")

    if "datetime" in df.columns:
        df["Date"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("Dataset does not contain a date/datetime column")

    df = df.dropna(subset=["Date"])
    df = df[df["Date"] >= pd.to_datetime(start)].copy()

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after load: {missing}")

    return df[needed].sort_values("Date").reset_index(drop=True)