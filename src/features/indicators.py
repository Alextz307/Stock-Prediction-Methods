import pandas as pd
from config.settings import settings


class TechnicalIndicators:
    """
    Optimized implementation of standard technical indicators (RSI, MACD, Bollinger Bands).
    Designed to work with pandas Series for vectorization.
    """

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI).
        
        Args:
            series (pd.Series): Price series.
            period (int): Lookback period.
        """

        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(
        series: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence (MACD).
        
        Returns:
            pd.DataFrame: Columns ['macd', 'macd_signal', 'macd_hist']
        """

        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame(
            {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
        )

    @staticmethod
    def bollinger_bands(
        series: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Bands.
        
        Returns:
            pd.DataFrame: Columns ['bb_upper', 'bb_mid', 'bb_lower']
        """
        
        mid = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper = mid + (std * std_dev)
        lower = mid - (std * std_dev)

        return pd.DataFrame({"bb_upper": upper, "bb_mid": mid, "bb_lower": lower})
