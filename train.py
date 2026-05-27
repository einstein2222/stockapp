import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import add_technical_features, add_target

HF_DATASET_PATH = "hf://datasets/usamaahmedsh/elliott-wave-market-data-complete/market_data_1d.parquet"

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_parquet(HF_DATASET_PATH)
    df.columns = [c.lower() for c in df.columns]

    if "source_category" in df.columns:
        df = df[df["source_category"].astype(str).str.lower() == "stocks"].copy()

    if "interval" in df.columns:
        df = df[df["interval"].astype(str).str.lower() == "1d"].copy()

    if "ticker" not in df.columns:
        raise ValueError("Dataset missing ticker column")

    df["Date"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "ticker": "Ticker",
    }
    df = df.rename(columns=rename_map)

    return df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)

def build_dataset():
    df = load_data()
    frames = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        if len(group) < 60:
            continue

        group = add_technical_features(group)
        group = add_target(group)
        frames.append(group)

    if not frames:
        raise ValueError("No usable tickers after preprocessing")

    return pd.concat(frames, ignore_index=True).fillna(0)

def train():
    df = build_dataset()

    feature_cols = [c for c in df.columns if c not in ["Date", "Target", "Ticker"]]

    X = df[feature_cols]
    y = df["Target"].astype(int)

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    baseline = LogisticRegression(max_iter=2000, random_state=42)
    baseline.fit(X_train_scaled, y_train)
    baseline_preds = baseline.predict(X_test_scaled)

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)

    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    print("\n=== Logistic Regression Baseline ===")
    print("Accuracy:", accuracy_score(y_test, baseline_preds))
    print("Precision:", precision_score(y_test, baseline_preds))
    print("Recall:", recall_score(y_test, baseline_preds))
    print("F1:", f1_score(y_test, baseline_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, baseline_preds))
    print(classification_report(y_test, baseline_preds))

    print("\n=== XGBoost Main Model ===")
    print("Accuracy:", accuracy_score(y_test, xgb_preds))
    print("Precision:", precision_score(y_test, xgb_preds))
    print("Recall:", recall_score(y_test, xgb_preds))
    print("F1:", f1_score(y_test, xgb_preds))
    print("ROC AUC:", roc_auc_score(y_test, xgb_probs))
    print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_preds))
    print(classification_report(y_test, xgb_preds))

    joblib.dump(xgb, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\nTraining complete.")

if __name__ == "__main__":
    train()