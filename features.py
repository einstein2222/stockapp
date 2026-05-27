import numpy as np
import pandas as pd
import ta

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)

    df["EMA_10"] = ta.trend.ema_indicator(df["Close"], window=10)
    df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_Diff"] = ta.trend.macd_diff(df["Close"])

    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Mid"] = bb.bollinger_mavg()

    df["Return_1D"] = df["Close"].pct_change()
    df["Return_5D"] = df["Close"].pct_change(5)
    df["Volatility_10D"] = df["Return_1D"].rolling(10).std()
    df["Price_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Close_to_Open"] = (df["Close"] - df["Open"]) / df["Open"]
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Momentum_5D"] = df["Close"] - df["Close"].shift(5)

    return df

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1].copy()
    df = df.dropna().reset_index(drop=True)
    return df