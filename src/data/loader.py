import yfinance as yf
import pandas as pd
import time

from src.features.volatility import VolatilityEstimator
from src.features.indicators import TechnicalIndicators
from config.settings import settings


def fetch_and_prepare_data(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetches market data from Yahoo Finance and enriches it with technical features.
    
    Args:
        ticker (str): Symbol to fetch.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        
    Returns:
        pd.DataFrame: Cleaned data with 'vol_*', 'rsi_*' features.
        
    Raises:
        ValueError: If no data is returned.
    """

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=20)
            if not df.empty:
                break
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
            
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to fetch data for {ticker} after {max_retries} attempts.")

            time.sleep(2 * (attempt + 1))
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except IndexError:
            pass
    
    df.columns = [c.lower() for c in df.columns]
    
    if "adj close" in df.columns:
        df = df.rename(columns={"adj close": "adj_close"})
    
    df = VolatilityEstimator.add_volatility_features(df)
    df["rsi_14"] = TechnicalIndicators.rsi(df["close"], period=settings.RSI_PERIOD)
    
    return df.dropna()
