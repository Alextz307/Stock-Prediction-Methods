import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from src.models.hybrid import HybridModel
from src.models.garch import GarchPredictor
from src.models.xgboost_classifier import DirectionalClassifier
from config.settings import settings


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        pass


class VolatilityTargetingStrategy(BaseStrategy):
    """
    A strategy that dynamically adjusts leverage based on forecasted volatility.
    
    Logic:
        - Forecast Annual Volatility (using GARCH-LSTM Hybrid).
        - Calculate Leverage = Target Vol / Forecast Vol.
        - Cap Leverage at 1.0 (no margin).
    """

    def __init__(self, target_vol: float = 0.15, trend_window: int = 100, ma_type: str = "sma", max_leverage: float | None = None, bearish_exposure: float = 0.0, hybrid_model: HybridModel | None = None, **kwargs):
        """
        Args:
            target_vol (float): Desired annualized volatility (e.g., 0.15 for 15%).
            trend_window (int): Lookback for trend filter (0 to disable).
            ma_type (str): 'sma' or 'ema'.
            max_leverage (float, optional): Maximum leverage allowed. Defaults to settings.MAX_LEVERAGE.
            bearish_exposure (float): Exposure scalar when trend is bearish (0.0 to 1.0).
            hybrid_model (HybridModel, optional): Pre-initialized model.
        """

        self.target_vol = target_vol
        self.trend_window = trend_window
        self.ma_type = ma_type.lower()
        self.max_leverage = max_leverage if max_leverage is not None else 1.5
        self.bearish_exposure = bearish_exposure
        self.model = hybrid_model if hybrid_model else HybridModel()
        self.lstm_params = kwargs

    def train(self, df: pd.DataFrame, trial: Any | None = None, **kwargs) -> None:
        """
        Trains the underlying Hybrid Volatility Model.
        """
      
        train_params = {**self.lstm_params, **kwargs}
        self.model.train(df, target_col="vol_garman_klass", trial=trial, **train_params)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates target position sizing (0.0 to 1.0).
        Includes Trend Filter: Price > MA(trend_window).
        """

        try:
            vol_forecast = self.model.predict_series(df)
            
            leverage = self.target_vol / vol_forecast
            signals = leverage.clip(lower=0.0, upper=self.max_leverage)
            
            if self.trend_window > 0:
                close = df['close']
                if self.ma_type == "ema":
                    trend_ma = close.ewm(span=self.trend_window, adjust=False).mean()
                else:
                    trend_ma = close.rolling(self.trend_window).mean()

                is_bullish = close > trend_ma
                
                signals = np.where(is_bullish, signals, signals * self.bearish_exposure)
                signals = pd.Series(signals, index=df.index)   
        except Exception as e:
            print(f"Hybrid Model Volatility check failed: {e}. Defaulting to Neutral.")
            signals = pd.Series(0, index=df.index)
            
        return signals


class AdaptiveBollingerStrategy(BaseStrategy):
    """
    Mean-reversion strategy using GARCH-adjusted Bollinger Bands.
    Includes a Trend Filter to prevent fighting strong trends.
    """

    def __init__(self, window: int = 20, k: float = 2.0, trend_window: int = 100, ma_type: str = "sma"):
        """
        Args:
            window (int): Moving average window for bands.
            k (float): Multiplier for GARCH volatility to set band width.
            trend_window (int): Period for the long-term trend MA.
            ma_type (str): 'sma' or 'ema'.
        """

        self.window = window
        self.k = k
        self.trend_window = trend_window
        self.ma_type = ma_type.lower()
        self.garch = GarchPredictor()

    def train(self, df: pd.DataFrame, trial: Any | None = None, **kwargs) -> None:
        """
        Trains the GARCH model on historical data.
        """

        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        self.garch.tune(returns)
        self.garch.fit(returns)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals (-1, 0, 1) using FIXED GARCH parameters.
        No Lookahead Bias.
        """
        
        close = df['close']
        returns = np.log(close / close.shift(1)).dropna()
        
        try:
            cond_vol = self.garch.generate_vol_series(returns)
        except ValueError:
            print("AdaptiveBollinger: GARCH not trained. Defaulting to historical std.")
            cond_vol = returns.rolling(20).std() * 100.0

        cond_vol = cond_vol.reindex(df.index).ffill().bfill()
        
        if self.ma_type == "ema":
            ma = close.ewm(span=self.window, adjust=False).mean()
            trend_ma = close.ewm(span=self.trend_window, adjust=False).mean()
        else:
            ma = close.rolling(self.window).mean()
            trend_ma = close.rolling(self.trend_window).mean()
            
        is_bull = close > trend_ma
        
        upper = ma + self.k * cond_vol
        lower = ma - self.k * cond_vol
                        
        long_entry = (is_bull) & (close < lower)
        long_exit = (is_bull) & (close > upper)
        
        short_entry = (~is_bull) & (close > upper)
        short_exit = (~is_bull) & (close < lower)
        
        final_signals = pd.Series(np.nan, index=df.index)
        
        final_signals[long_entry] = 1
        final_signals[long_exit] = 0
        
        final_signals[short_entry] = -1
        final_signals[short_exit] = 0 
        
        final_signals = final_signals.ffill().fillna(0)
        
        return final_signals


class MomentumGatekeeperStrategy(BaseStrategy):
    """
    Momentum strategy enhanced with Machine Learning (XGBoost).
    
    Logic:
        - Only enter Long if:
            1. Trend is Bullish (Price > SMA).
            2. XGBoost Classifier predicts a high probability of Up move.
    """

    def __init__(self, ma_window: int = 50, prob_threshold: float = 0.55, **kwargs):
        """
        Args:
            ma_window (int): Trend filter window.
            prob_threshold (float): Confidence threshold for XGBoost (0.5 to 1.0).
            **kwargs: Hyperparameters passed to the internal XGBoost Classifier.
        """

        self.ma_window = ma_window
        self.threshold = prob_threshold
        self.classifier = DirectionalClassifier(params=kwargs)
        self.fitted = False

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering for the classifier.
        """

        df = df.copy()
        
        df['vol_20'] = df['return_1d'].rolling(20).std()
        df['ma_ratio'] = df['close'] / df['close'].rolling(self.ma_window).mean()
        
        features = settings.XGB_FEATURES
        return df[features].fillna(0)

    def train(self, df: pd.DataFrame, trial: Any | None = None, **kwargs) -> None:
        """
        Trains the internal XGBoost classifier.
        """
        
        if "ma_window" in kwargs:
            self.ma_window = kwargs.pop("ma_window")
        if "prob_threshold" in kwargs:
            self.threshold = kwargs.pop("prob_threshold")

        if kwargs:
            self.classifier.set_params(**kwargs)

        print("Training MomentumGatekeeperStrategy (XGBoost)...")
        X = self._get_features(df)
        
        y = (df['close'].shift(-1) > df['close']).astype(int)
        
        X = X.iloc[:-1]
        y = y.iloc[:-1]
        
        self.classifier.fit(X, y)
        self.fitted = True

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates buy (1) signals based on Trend + ML Confidence.
        """
        
        ma = df['close'].rolling(self.ma_window).mean()
        trend_bullish = df['close'] > ma
        
        probs = pd.Series(0.0, index=df.index)
        if self.fitted:
            X = self._get_features(df)
            probs = self.classifier.predict_proba(X)
        
        signals = pd.Series(0, index=df.index)
        long_condition = (trend_bullish) & (probs > self.threshold)
        signals[long_condition] = 1
        
        return signals
