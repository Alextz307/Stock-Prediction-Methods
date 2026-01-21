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

    def optimize(self, strategy_name: str, n_trials: int = 50) -> dict[str, Any]:
        """
        Runs the optimization loop.
        
        Args:
            strategy_name (str): 'VolTarget', 'Bollinger', or 'MomentumXGB'.
            n_trials (int): Number of trials.
            
        Returns:
            dict: Best parameters found.
        """

        logger.info(f"Optimizing {strategy_name} for {self.ticker} ({n_trials} trials)...")
        
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        study.optimize(lambda trial: self._objective(trial, strategy_name), n_trials=n_trials)
        
        logger.info(f"[{self.ticker}] Best params for {strategy_name}: {study.best_params}")
        return study.best_params

    def _objective(self, trial: optuna.Trial, strategy_name: str) -> float:
        params = self._suggest_params(trial, strategy_name)
        
        strategy = self._get_strategy_instance(strategy_name, params)
        
        try:
            if hasattr(strategy, "train"):
                strategy.train(self.train_df, trial=trial, **params)
            
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
            
            logger.info(f"[{self.ticker}] Trial {trial.number} finished with value: {sharpe:.4f} and parameters: {params}")
            return sharpe
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return -999.0

    def _suggest_params(self, trial: optuna.Trial, strategy_name: str) -> dict[str, Any]:
        if strategy_name == "VolTarget":
            return {
                "target_vol": trial.suggest_float("target_vol", 0.05, 0.40, step=0.01),
                "trend_window": trial.suggest_int("trend_window", 5, 252),
                "ma_type": trial.suggest_categorical("ma_type", ["sma", "ema"]),
                "max_leverage": trial.suggest_float("max_leverage", 1.0, 3.0, step=0.1),
                "bearish_exposure": trial.suggest_float("bearish_exposure", 0.0, 0.6, step=0.05),
                "lstm_hidden_dim": trial.suggest_categorical("lstm_hidden_dim", [32, 64, 128]),
                "lstm_lr": trial.suggest_float("lstm_lr", 1e-4, 1e-2, log=True),
                "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
                "lstm_lookback": trial.suggest_int("lstm_lookback", 10, 60, step=5),
                "lstm_dropout": trial.suggest_float("lstm_dropout", 0.1, 0.5, step=0.1)
            }
        elif strategy_name == "Bollinger":
            return {
                "window": trial.suggest_int("window", 10, 50),
                "k": trial.suggest_float("k", 1.0, 3.0, step=0.1),
                "trend_window": trial.suggest_int("trend_window", 5, 252),
                "ma_type": trial.suggest_categorical("ma_type", ["sma", "ema"])
            }
        elif strategy_name == "MomentumXGB":
            return {
                "ma_window": trial.suggest_int("ma_window", 10, 100),
                "prob_threshold": trial.suggest_float("prob_threshold", 0.51, 0.70, step=0.01),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1)
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
