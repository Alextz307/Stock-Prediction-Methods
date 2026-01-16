import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

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

    def train(self, df: pd.DataFrame, target_col: str = "vol_garman_klass") -> None:
        """
        Trains the Hybrid Model pipeline.
        
        Steps:
            1. Tune & Fit GARCH on log-returns.
            2. Calculate GARCH residuals (Actual Vol - GARCH Forecast).
            3. Train LSTM to predict these residuals using market features.
            
        Args:
            df (pd.DataFrame): Training data.
            target_col (str): Column name for the "Ground Truth" volatility (e.g., Garman-Klass).
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

        dataset = FinancialDataset(scaled_df, target_col="residuals")
        loader = DataLoader(dataset, batch_size=settings.LSTM_BATCH_SIZE, shuffle=True)

        input_dim = dataset[0][0].shape[1]
        self.lstm = MarketLSTM(input_dim=input_dim)

        optimizer = optim.Adam(self.lstm.parameters(), lr=settings.LSTM_LEARNING_RATE)
        criterion = nn.MSELoss()

        self.lstm.train()
        for epoch in range(settings.LSTM_EPOCHS):
            total_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()

                preds = self.lstm(X_batch)
                loss = criterion(preds.squeeze(), y_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{settings.LSTM_EPOCHS} - Loss: {total_loss / len(loader):.6f}"
                )

    def predict(self, recent_data: pd.DataFrame, horizon: int = 1) -> float:
        """
        Predicts volatility for the next time step (Live Inference).
        """
        if self.lstm is None:
            raise ValueError("Model not trained yet.")

        garch_daily_forecast = self.garch.predict_volatility(horizon)
        garch_forecast = garch_daily_forecast * float(np.sqrt(252))

        if len(recent_data) < settings.LSTM_LOOKBACK:
            raise ValueError(f"Not enough data. Need {settings.LSTM_LOOKBACK}.")

        recent_window = recent_data.tail(settings.LSTM_LOOKBACK).copy()

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
        No Lookahead Bias.
        """
        if self.lstm is None:
            raise ValueError("Model not trained yet.")
        
        # 1. GARCH Forecasts (Fixed Params)
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        garch_vol = self.garch.generate_vol_series(returns)
        garch_vol_annual = (garch_vol / 100.0) * np.sqrt(252)
        
        # Realign to DF
        garch_vol_aligned = garch_vol_annual.reindex(df.index).ffill().bfill()
        
        # 2. LSTM Residuals (Eval Mode)
        ignore_cols = ["ticker", "date", "target_direction", "return_1d", "residuals"]
        feature_cols = [c for c in df.columns if c not in ignore_cols]
        
        # Prepare Data
        # Ensure we have the same features
        X_raw = df[feature_cols].values
        X_scaled = self.scaler.transform(X_raw)
        
        # We need sequences for LSTM. This is slow iteratively, 
        # but for backtesting we can use a Rolling Dataset or simple iteration.
        # For speed in vectorization, we might just assume the LSTM correction 
        # is minor or run it batch-wise if possible.
        # Given LSTM needs lookback window, let's create a dataset.
        
        from src.models.dataset import FinancialDataset
        from torch.utils.data import DataLoader
        
        # Create a dummy target just for Dataset class compatibility
        dummy_df = df.copy()
        dummy_df["residuals"] = 0.0 
        dummy_df[feature_cols] = X_scaled # Use scaled values directly if modifying Dataset logic? 
        # actually FinancialDataset scales externally usually. 
        # Let's check logic: FinancialDataset takes raw or scaled? 
        # It takes df. In train() we passed scaled_df. So yes.
        
        dummy_df[feature_cols] = X_scaled
        
        ds = FinancialDataset(dummy_df, target_col="residuals") # Uses LSTM_LOOKBACK inside
        dl = DataLoader(ds, batch_size=1024, shuffle=False)
        
        lstm_preds = []
        self.lstm.eval()
        with torch.no_grad():
            for X_batch, _ in dl:
                preds = self.lstm(X_batch)
                lstm_preds.extend(preds.numpy().flatten())
                
        # LSTM predictions start after LOOKBACK
        # Pad the beginning with 0
        padding = [0.0] * settings.LSTM_LOOKBACK
        lstm_series = np.array(padding + lstm_preds)
        
        # Handle length mismatch if any (Dataset usually trims)
        # FinancialDataset length = len(df) - lookback
        # So padding restores it to len(df).
        
        if len(lstm_series) < len(df):
            # Pad end if needed? No, usually start.
            pass
        elif len(lstm_series) > len(df):
            lstm_series = lstm_series[:len(df)]
            
        lstm_series_pd = pd.Series(lstm_series, index=df.index)
        
        final_vol = garch_vol_aligned + lstm_series_pd
        return final_vol.clip(lower=0.001)
