import optuna
import pandas as pd
from typing import Any
import logging

from src.strategies.strategies import (
    BaseStrategy,
    VolatilityTargetingStrategy,
    AdaptiveBollingerStrategy,
    MomentumGatekeeperStrategy
)
from src.strategies.metrics import calculate_metrics
from config.settings import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyTuner:
    """
    Optimizes strategy parameters using Optuna.
    """

    def __init__(self, ticker: str, df: pd.DataFrame):
        """
        Args:
            ticker (str): The ticker symbol being optimized.
            df (pd.DataFrame): The historical data available for tuning (should exclude the final backtest test set).
        """

        self.ticker = ticker
        self.df = df
        
        split_idx = int(len(df) * 0.8)
        self.train_df = df.iloc[:split_idx]
        self.val_df = df.iloc[split_idx:]
        
        self.default_transaction_fee = settings.BACKTEST_TRANSACTION_FEE

    def optimize(self, strategy_name: str, n_trials: int = 20) -> dict[str, Any]:
        """
        Runs the optimization loop.
        
        Args:
            strategy_name (str): 'VolTarget', 'Bollinger', or 'MomentumXGB'.
            n_trials (int): Number of trials.
            
        Returns:
            dict: Best parameters found.
        """

        logger.info(f"Optimizing {strategy_name} for {self.ticker} ({n_trials} trials)...")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self._objective(trial, strategy_name), n_trials=n_trials)
        
        logger.info(f"Best params for {strategy_name} on {self.ticker}: {study.best_params}")
        return study.best_params

    def _objective(self, trial: optuna.Trial, strategy_name: str) -> float:
        params = self._suggest_params(trial, strategy_name)
        
        strategy = self._get_strategy_instance(strategy_name, params)
        
        try:
            if hasattr(strategy, "train"):
                strategy.train(self.train_df)
            
            full_signals = strategy.generate_signals(self.df)
            
            val_signals = full_signals.loc[self.val_df.index]
            val_returns_asset = self.val_df["close"].pct_change().dropna()
            
            aligned_signals = val_signals.shift(1).dropna()
            common_idx = aligned_signals.index.intersection(val_returns_asset.index)
            
            aligned_signals = aligned_signals.loc[common_idx]
            actual_returns = val_returns_asset.loc[common_idx]
            
            raw_returns = aligned_signals * actual_returns
            turnover = aligned_signals.diff().abs().fillna(0)
            costs = turnover * self.default_transaction_fee
            
            net_returns = raw_returns - costs
            
            metrics = calculate_metrics(net_returns)
            sharpe = metrics.get("Sharpe Ratio", -999.0)
            
            return sharpe
        except Exception as e:
            return -999.0

    def _suggest_params(self, trial: optuna.Trial, strategy_name: str) -> dict[str, Any]:
        if strategy_name == "VolTarget":
            return {
                "target_vol": trial.suggest_float("target_vol", 0.05, 0.40, step=0.01),
                "trend_window": trial.suggest_int("trend_window", 0, 252),
                "ma_type": trial.suggest_categorical("ma_type", ["sma", "ema"])
            }
        elif strategy_name == "Bollinger":
            return {
                "window": trial.suggest_int("window", 10, 50),
                "k": trial.suggest_float("k", 1.0, 3.0, step=0.1),
                "trend_window": trial.suggest_int("trend_window", 50, 252),
                "ma_type": trial.suggest_categorical("ma_type", ["sma", "ema"])
            }
        elif strategy_name == "MomentumXGB":
            return {
                "ma_window": trial.suggest_int("ma_window", 10, 100),
                "prob_threshold": trial.suggest_float("prob_threshold", 0.51, 0.70, step=0.01)
            }
            
        return {}

    def _get_strategy_instance(self, strategy_name: str, params: dict[str, Any]) -> BaseStrategy:
        if strategy_name == "VolTarget":
            return VolatilityTargetingStrategy(**params)
        elif strategy_name == "Bollinger":
            return AdaptiveBollingerStrategy(**params)
        elif strategy_name == "MomentumXGB":
            return MomentumGatekeeperStrategy(**params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
