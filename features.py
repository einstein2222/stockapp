import pandas as pd
import ta


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], 10)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], 20)

    df["EMA_10"] = ta.trend.ema_indicator(df["Close"], 10)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], 20)

    df["RSI_14"] = ta.momentum.rsi(df["Close"], 14)

    df["MACD"] = ta.trend.macd_diff(df["Close"])

    bb = ta.vgrep -R "add_features" .olatility.BollingerBands(df["Close"], window=20)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()

    df["Return_1D"] = df["Close"].pct_change()
    df["Volatility_10D"] = df["Return_1D"].rolling(10).std()

    df["Price_Range"] = (df["High"] - df["Low"]) / df["Close"]

    return df


def encode_ticker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dummies = pd.get_dummies(df["Ticker"], prefix="Ticker")
    return pd.concat([df.drop(columns=["Ticker"]), dummies], axis=1)