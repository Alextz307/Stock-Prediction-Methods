from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Backtest Configuration ---

    BACKTEST_TICKERS: list[str] = [
        "SPY", "AAPL", "MSFT", "GOOG", "NVDA", "AMZN", "META", "TSLA", "KO", "PG", "XOM"
    ]
    BACKTEST_TEST_DAYS: int = 252
    BACKTEST_HISTORY_BUFFER_DAYS: int = 5 * 365
    BACKTEST_TRANSACTION_FEE: float = 0.001
    
    
    # --- Strategy Parameters ---

    # Volatility Targeting
    VOL_TARGET: float = 0.15
    
    # Adaptive Bollinger
    BOLLINGER_WINDOW: int = 20
    BOLLINGER_K: float = 2.0
    BOLLINGER_TREND_WINDOW: int = 200
    
    # Momentum Gatekeeper
    MOMENTUM_MA_WINDOW: int = 50
    MOMENTUM_PROB_THRESHOLD: float = 0.55
    
    # XGBoost Hyperparameters
    XGB_N_ESTIMATORS: int = 100
    XGB_LEARNING_RATE: float = 0.05
    XGB_MAX_DEPTH: int = 5


    # --- Feature Parameters ---
    RSI_PERIOD: int = 14
    
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD_DEV: float = 2.0
    
    VOLATILITY_WINDOW: int = 22
    VOLATILITY_ANNUALIZATION: int = 252
    
    TUNED_PARAMS_FILE: str = "config/tuned_params.json"

    GARCH_MAX_P: int = 5
    GARCH_MAX_Q: int = 5
    GARCH_DISTRIBUTION: Literal[
        "normal",
        "gaussian",
        "t",
        "studentst",
        "skewstudent",
        "skewt",
        "ged",
        "generalized error",
    ] = "skewt" 

    LSTM_INPUT_DIM: int = 0
    LSTM_HIDDEN_DIM: int = 64
    LSTM_NUM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0.2
    LSTM_LEARNING_RATE: float = 0.001
    LSTM_EPOCHS: int = 100
    LSTM_BATCH_SIZE: int = 32
    LSTM_LOOKBACK: int = 30 

    class Config:
        env_file = ".env"


settings = Settings()
