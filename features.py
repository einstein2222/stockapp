import pandas as pd
import ta

def add__technical_features(df):
    df = df.copy()

    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], 10)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], 20)

    df["EMA_10"] = ta.trend.ema_indicator(df["Close"], 10)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], 20)

    df["RSI"] = ta.momentum.rsi(df["Close"], 14)

    df["MACD"] = ta.trend.macd_diff(df["Close"])

    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df