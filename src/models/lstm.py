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

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initializes the LSTM model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output neurons (default 1).
            hidden_dim (int): LSTM hidden dimension.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout probability.
        """

        super(MarketLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        lstm_dropout_arg = dropout if self.num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout_arg,
        )
        
        self.dropout = nn.Dropout(dropout)

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
        
        last_step = self.dropout(last_step)
        
        prediction = self.fc(last_step)
        return cast(torch.Tensor, prediction)
