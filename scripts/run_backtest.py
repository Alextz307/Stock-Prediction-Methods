import warnings

from src.engine.core import BacktestEngine
from src.strategies.strategies import (
    VolatilityTargetingStrategy,
    AdaptiveBollingerStrategy,
    MomentumGatekeeperStrategy
)
from config.settings import settings


warnings.filterwarnings("ignore")


def main() -> None:
    strategies = {
        "VolTarget": VolatilityTargetingStrategy(),
        "Bollinger": AdaptiveBollingerStrategy(),
        "MomentumXGB": MomentumGatekeeperStrategy(),
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

