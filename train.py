# train.py
import joblib
from pathlib import Path
import json
import pandas as pd
from xgboost import XGBClassifier

from data_loader import download_stock_data
from features import add_technical_features

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


def build_dataset():
    frames = []

    for t in TICKERS:
        df = download_stock_data(t)
        df = add_technical_features(df)

        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df = df.dropna().reset_index(drop=True)

        frames.append(df)

    return pd.concat(frames, ignore_index=True).fillna(0)


def train():
    df = build_dataset()

    feature_cols = [c for c in df.columns if c not in ["Target", "Date", "Ticker"]]

    X = df[feature_cols]
    y = df["Target"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    train()