import numpy as np
import pandas as pd


class VolatilityEstimator:
    """
    Implements range-based volatility estimators (Garman-Klass, Parkinson).
    These are generally more efficient than simple Close-to-Close standard deviation.
    """
    
    @staticmethod
    def garman_klass(
        high: pd.Series,
        low: pd.Series,
        open_p: pd.Series,
        close: pd.Series,
        window: int = 22,
        trading_periods: int = 252,
    ) -> pd.Series:
        """
        Calculates the Garman-Klass volatility estimator.
        
        Formula:
            0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
            
        Args:
            window (int): Rolling window for smoothing.
            trading_periods (int): Annualization factor (typically 252).
            
        Returns:
            pd.Series: Annualized volatility series.
        """

        epsilon = 1e-8

        log_hl = np.log(high / (low + epsilon))
        log_co = np.log(close / (open_p + epsilon))

        var_daily = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)

        vol_series = np.sqrt(var_daily.rolling(window=window).mean()) * np.sqrt(
            trading_periods
        )

        return vol_series 

    @staticmethod
    def parkinson(
        high: pd.Series, low: pd.Series, window: int = 22, trading_periods: int = 252
    ) -> pd.Series:
        """
        Calculates the Parkinson volatility estimator (High-Low).
        
        Formula:
            (1 / 4*ln(2)) * ln(H/L)^2
        """

        epsilon = 1e-8
        log_hl = np.log(high / (low + epsilon))
        var_daily = (1.0 / (4.0 * np.log(2.0))) * (log_hl**2)

        return np.sqrt(var_daily.rolling(window=window).mean()) * np.sqrt( 
            trading_periods
        )

    @staticmethod
    def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'vol_garman_klass', 'vol_parkinson', and 'vol_close_close' to the DataFrame.
        Handles multi-ticker grouping if 'ticker' column is present.
        """

        df = df.copy()

        if "ticker" in df.columns:
            df["vol_garman_klass"] = df.groupby("ticker", group_keys=False)[
                ["high", "low", "open", "close"]
            ].apply(
                lambda x: VolatilityEstimator.garman_klass(
                    x["high"], x["low"], x["open"], x["close"]
                )
            )
            df["vol_parkinson"] = df.groupby("ticker", group_keys=False)[
                ["high", "low"]
            ].apply(lambda x: VolatilityEstimator.parkinson(x["high"], x["low"]))

            df["vol_close_close"] = df.groupby("ticker", group_keys=False)[
                "close"
            ].apply(lambda x: x.pct_change().rolling(22).std() * np.sqrt(252))
        else:
            df["vol_garman_klass"] = VolatilityEstimator.garman_klass(
                df["high"], df["low"], df["open"], df["close"]
            )
            df["vol_parkinson"] = VolatilityEstimator.parkinson(df["high"], df["low"])

        return df.dropna()
