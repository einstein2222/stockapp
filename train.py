import json
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from data_loader import download_stock_data
from features import add_features

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

def build():
    frames = []

    for t in TICKERS:
        df = download_stock_data(t)
        df = add_features(df)
        df["Ticker"] = t
        frames.append(df)

    df = __import__("pandas").concat(frames).fillna(0)

    features = [c for c in df.columns if c not in ["Target", "Date"]]

    X = df[features]
    y = df["Target"]

    log = LogisticRegression(max_iter=2000)
    log.fit(X, y)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    xgb.fit(X, y)

    joblib.dump(xgb, MODEL_DIR / "model.pkl")
    joblib.dump(features, MODEL_DIR / "features.json")

    print("Training complete.")

if __name__ == "__main__":
    build()