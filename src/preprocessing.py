import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date")
    rsi = RSIIndicator(close=df["Close"], window=14)
    macd = MACD(close=df["Close"])
    df["RSI"] = rsi.rsi()
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    return df.dropna().reset_index(drop=True)
