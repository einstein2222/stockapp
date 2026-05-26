import json
import time
from pathlib import Path

import pandas as pd
import requests

API_KEY = "8RQ53JQR9YALUWED"
CACHE_DIR = Path(__file__).resolve().parent / "av_cache"
CACHE_DIR.mkdir(exist_ok=True)

def download_stock_data(ticker: str, start="2020-01-01"):
    ticker = ticker.upper().strip()
    cache_file = CACHE_DIR / f"{ticker}.json"

    if cache_file.exists():
        with open(cache_file, "r") as f:
            data = json.load(f)
    else:
        url = (
            "https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY"
            f"&symbol={ticker}"
            f"&outputsize=compact"
            f"&apikey={API_KEY}"
        )

        r = requests.get(url, timeout=30)
        data = r.json()

        if "Time Series (Daily)" not in data:
            raise ValueError(f"API error: {data}")

        with open(cache_file, "w") as f:
            json.dump(data, f)

        time.sleep(1.1)  # stay under the free per-second limit

    ts = data["Time Series (Daily)"]

    rows = []
    for date, values in ts.items():
        rows.append({
            "Date": date,
            "Open": float(values["1. open"]),
            "High": float(values["2. high"]),
            "Low": float(values["3. low"]),
            "Close": float(values["4. close"]),
            "Volume": float(values["5. volume"]),
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= pd.to_datetime(start)]
    df = df.sort_values("Date").reset_index(drop=True)
    return df