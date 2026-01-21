from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BACKTEST_TICKERS: list[str] = [
        "SPY", "AAPL", "MSFT", "GOOG", "NVDA", "AMZN", "META", "TSLA", "KO", "PG", "XOM", "BAC", "CMG"
    ]
    BACKTEST_TEST_DAYS: int = 252
    BACKTEST_HISTORY_BUFFER_DAYS: int = 6 * 365
    BACKTEST_TRANSACTION_FEE: float = 0.001

    INITIAL_CAPITAL: float = 10000.0
    

    XGB_FEATURES: list[str] = ["return_1d", "return_5d", "return_21d", "vol_20", "ma_ratio", "rsi_14", "macd", "macd_signal", "macd_hist", "volume"]
    
    TUNING_N_TRIALS: int = 1024
    TUNED_PARAMS_FILE: str = "config/tuned_params.json"

    class Config:
        env_file = ".env"


settings = Settings()
