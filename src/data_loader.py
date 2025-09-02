import os
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

def load_stock_data(ticker: str, years: int = 5, fallback_csv: str = None) -> pd.DataFrame:
    try:
        end = date.today()
        start = end - timedelta(days=years*365)
        df = yf.download(ticker, start=start, end=end)
        if df is None or df.empty:
            raise RuntimeError("Empty data")
        return df.reset_index()
    except Exception:
        if fallback_csv and os.path.exists(fallback_csv):
            return pd.read_csv(fallback_csv, parse_dates=["Date"])
        raise
