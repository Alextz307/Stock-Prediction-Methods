import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from config.settings import settings


class FinancialDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        lookback: int = settings.LSTM_LOOKBACK,
    ):
        self.lookback = lookback

        ignore_cols = ["ticker", "date", "target_direction", "return_1d", target_col]

        feature_cols = [c for c in data.columns if c not in ignore_cols]

        self.X = data[feature_cols].values.astype(np.float32)
        self.y = data[target_col].values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X) - self.lookback

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_window = self.X[idx : idx + self.lookback]
        y_label = self.y[idx + self.lookback]
        return torch.tensor(x_window), torch.tensor(y_label)
