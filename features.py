import pandas as pd
import ta


def add_technical_features(df):
    df = df.copy()

    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)

    df["EMA_10"] = ta.trend.ema_indicator(df["Close"], window=10)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

    df["MACD"] = ta.trend.macd_diff(df["Close"])

    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df.dropna().reset_index(drop=True)


def encode_ticker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dummies = pd.get_dummies(df["Ticker"], prefix="Ticker")
    return pd.concat([df.drop(columns=["Ticker"]), dummies], axis=1)