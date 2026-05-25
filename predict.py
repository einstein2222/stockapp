from pathlib import Path
import json
import joblib

from data_loader import download_stock_data
from features import add_technical_features, encode_ticker
from sentiment import get_news_sentiment


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "saved_models"

MODEL = joblib.load(MODEL_DIR / "xgb_model.pkl")

with open(MODEL_DIR / "feature_columns.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)


def predict_ticker(ticker: str):
    ticker = ticker.upper().strip()

    df = download_stock_data(ticker)
    df["Ticker"] = ticker

    df = add_technical_features(df)
    df = df.dropna()

    latest = df.iloc[-1:].copy()
    latest = encode_ticker(latest)

    X = latest.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    proba_up = float(MODEL.predict_proba(X)[0, 1])

    sentiment_score, headlines = get_news_sentiment(ticker)

    sentiment_adj = (sentiment_score + 1) / 2
    combined = 0.85 * proba_up + 0.15 * sentiment_adj

    return {
        "ticker": ticker,
        "prediction": int(proba_up > 0.5),
        "proba_up": proba_up,
        "proba_down": 1 - proba_up,
        "sentiment_score": sentiment_score,
        "combined_score": combined,
        "headlines": headlines[:5],
    }