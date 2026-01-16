import warnings

from src.engine.core import BacktestEngine
from src.strategies.strategies import (
    VolatilityTargetingStrategy,
    AdaptiveBollingerStrategy,
    MomentumGatekeeperStrategy
)
from config.settings import settings


warnings.filterwarnings("ignore")


def main():
    strategies = {
        "VolTarget": VolatilityTargetingStrategy(target_vol=settings.VOL_TARGET),
        "Bollinger": AdaptiveBollingerStrategy(
            window=settings.BOLLINGER_WINDOW, 
            k=settings.BOLLINGER_K,
            trend_window=settings.BOLLINGER_TREND_WINDOW
        ),
        "MomentumXGB": MomentumGatekeeperStrategy(
            ma_window=settings.MOMENTUM_MA_WINDOW, 
            prob_threshold=settings.MOMENTUM_PROB_THRESHOLD
        ),
    }
    
    engine = BacktestEngine(
        strategies=strategies,
        tickers=settings.BACKTEST_TICKERS,
        test_days=settings.BACKTEST_TEST_DAYS,
        results_dir="results"
    )
    
    engine.run()


if __name__ == "__main__":
    main()

