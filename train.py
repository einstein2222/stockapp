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

from data_loader import download_stock_data
from features import add_technical_features, add_target

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "NFLX", "JPM", "XOM",
    "UNH", "COST"
]

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


def build_datasets():
    train_frames = []
    test_frames = []

    for t in TICKERS:
        df = download_stock_data(t, allow_download=True)
        df = add_technical_features(df)
        df = add_target(df)

        split_idx = int(len(df) * 0.8)
        train_frames.append(df.iloc[:split_idx].copy())
        test_frames.append(df.iloc[split_idx:].copy())

    train_df = pd.concat(train_frames, ignore_index=True).fillna(0)
    test_df = pd.concat(test_frames, ignore_index=True).fillna(0)

    return train_df, test_df


def train():
    train_df, test_df = build_datasets()

    feature_cols = [
        c for c in train_df.columns
        if c not in ["Date", "Target"]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["Target"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["Target"].astype(int)

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

    print(f"\nSaved model artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    train()