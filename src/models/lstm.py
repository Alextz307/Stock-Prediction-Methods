import torch
import torch.nn as nn
from typing import cast

from config.settings import settings


class MarketLSTM(nn.Module):
    """
    Standard LSTM for financial time series prediction.
    
    Architecture:
        Input -> LSTM Layers -> Dropout -> Fully Connected -> Output
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        """
        Initializes the LSTM model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output neurons (default 1).
        """

        super(MarketLSTM, self).__init__()

        self.hidden_dim = settings.LSTM_HIDDEN_DIM
        self.num_layers = settings.LSTM_NUM_LAYERS

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=settings.LSTM_DROPOUT,
        )

        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Prediction for the last time step.
        """
        
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        prediction = self.fc(last_step)
        return cast(torch.Tensor, prediction)
