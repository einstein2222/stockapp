import json
import joblib
from pathlib import Path

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
        df["Ticker"] = t

        df = add_technical_features(df)

        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        df = df.dropna()
        frames.append(df)

    return __import__("pandas").concat(frames).fillna(0)


def train():
    df = build_dataset()

    feature_cols = [c for c in df.columns if c not in ["Target", "Date"]]

    X = df[feature_cols]
    y = df["Target"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.json")

    print("Training complete.")


if __name__ == "__main__":
    train()