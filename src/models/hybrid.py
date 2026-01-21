import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Any
import optuna

from config.settings import settings
from src.models.garch import GarchPredictor
from src.models.lstm import MarketLSTM
from src.models.dataset import FinancialDataset


class HybridModel:
    """
    Implements a Hybrid Volatility Model: GARCH + LSTM.
    
    1. GARCH: Captures linear, autoregressive volatility.
    2. LSTM: Captures non-linear residuals (errors) of the GARCH model.
    """

    def __init__(self) -> None:
        self.garch = GarchPredictor()
        self.lstm: MarketLSTM | None = None
        self.scaler = StandardScaler()
        self.lookback = 30

    def train(self, df: pd.DataFrame, target_col: str = "vol_garman_klass", trial: Any | None = None, **kwargs) -> None:
        """
        Trains the Hybrid Model pipeline.
        
        Steps:
            1. Tune & Fit GARCH on log-returns.
            2. Calculate GARCH residuals (Actual Vol - GARCH Forecast).
            3. Train LSTM to predict these residuals using market features.
            
        Args:
            df (pd.DataFrame): Training data.
            target_col (str): "Ground Truth" volatility column.
            trial (optuna.Trial, optional): Optuna trial for pruning.
            **kwargs: Hyperparameters for LSTM (hidden_dim, lr, num_layers).
        """

        print("--- Starting Hybrid Model Training ---")

        close_series = df["close"]
        returns_array = np.log(close_series / close_series.shift(1))
        returns = pd.Series(returns_array, index=df.index).dropna()

        print("Stage 1: Fitting GARCH...")
        self.garch.tune(returns)
        self.garch.fit(returns)

        garch_vol = self.garch.model_res.conditional_volatility

        aligned_df = df.loc[garch_vol.index].copy()
        realized_vol = aligned_df[target_col]

        garch_vol_annualized = (garch_vol / 100.0) * np.sqrt(252)

        residuals = realized_vol - garch_vol_annualized
        aligned_df["residuals"] = residuals

        print(f"Residuals Mean: {residuals.mean():.6f}, Std: {residuals.std():.6f}")

        print("Stage 2: Training LSTM on Residuals...")

        ignore_cols = ["ticker", "date", "target_direction", "return_1d", "residuals"]
        feature_cols = [c for c in aligned_df.columns if c not in ignore_cols]

        X_raw = aligned_df[feature_cols].values
        self.scaler.fit(X_raw)

        scaled_df = aligned_df.copy()
        scaled_df[feature_cols] = self.scaler.transform(X_raw)

        input_dim = scaled_df[feature_cols].shape[1]
        
        hidden_dim = kwargs.get("lstm_hidden_dim", 64)
        num_layers = kwargs.get("lstm_layers", 2)
        lr = kwargs.get("lstm_lr", 0.001)
        dropout = kwargs.get("lstm_dropout", 0.2)
        self.lookback = kwargs.get("lstm_lookback", 30)
        
        dataset = FinancialDataset(scaled_df, target_col="residuals", lookback=self.lookback)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.lstm = MarketLSTM(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout
        )

        optimizer = optim.Adam(self.lstm.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.lstm.train()
        for epoch in range(100):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()

                preds = self.lstm(X_batch)
                loss = criterion(preds.view(-1), y_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/100 - Loss: {avg_loss:.6f}")
                
            if trial:
                trial.report(avg_loss, epoch)
                
                if trial.should_prune():
                    print(f"Trial pruned at epoch {epoch}")
                    raise optuna.TrialPruned()

    def predict(self, recent_data: pd.DataFrame, horizon: int = 1) -> float:
        """
        Predicts volatility for the next time step (Live Inference).
        """

        if self.lstm is None:
            raise ValueError("Model not trained yet.")

        garch_daily_forecast = self.garch.predict_volatility(horizon)
        garch_forecast = garch_daily_forecast * float(np.sqrt(252))

        if len(recent_data) < self.lookback:
            raise ValueError(f"Not enough data. Need {self.lookback}.")

        recent_window = recent_data.tail(self.lookback).copy()

        ignore_cols = ["ticker", "date", "target_direction", "return_1d", "residuals"]
        feature_cols = [c for c in recent_data.columns if c not in ignore_cols]

        X_raw = recent_window[feature_cols].values
        X_scaled = self.scaler.transform(X_raw) 
        X_tensor = torch.tensor(X_scaled.astype(np.float32)).unsqueeze(0)

        self.lstm.eval()
        with torch.no_grad():
            pred_tensor = self.lstm(X_tensor)
            lstm_residual = float(pred_tensor.item())

        final_vol = garch_forecast + lstm_residual
        return max(0.001, final_vol)

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates volatility forecasts for an entire dataframe using FIXED parameters.
        """

        if self.lstm is None:
            raise ValueError("Model not trained yet.")
        
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        garch_vol = self.garch.generate_vol_series(returns)
        garch_vol_annual = (garch_vol / 100.0) * np.sqrt(252)
        
        garch_vol_aligned = garch_vol_annual.reindex(df.index).ffill().bfill()
        
        ignore_cols = ["ticker", "date", "target_direction", "return_1d", "residuals"]
        feature_cols = [c for c in df.columns if c not in ignore_cols]
        
        X_raw = df[feature_cols].values
        X_scaled = self.scaler.transform(X_raw)
        
        dummy_df = df.copy()
        dummy_df["residuals"] = 0.0 
        dummy_df[feature_cols] = X_scaled
        
        ds = FinancialDataset(dummy_df, target_col="residuals", lookback=self.lookback)
        dl = DataLoader(ds, batch_size=1024, shuffle=False)
        
        lstm_preds = []
        self.lstm.eval()
        with torch.no_grad():
            for X_batch, _ in dl:
                preds = self.lstm(X_batch)
                lstm_preds.extend(preds.numpy().flatten())
                
        padding = [0.0] * self.lookback
        lstm_series = np.array(padding + lstm_preds)
        
        if len(lstm_series) > len(df):
            lstm_series = lstm_series[:len(df)]
            
        lstm_series_pd = pd.Series(lstm_series, index=df.index)
        
        final_vol = garch_vol_aligned + lstm_series_pd
        return final_vol.clip(lower=0.001)
