import json
import os
from datetime import datetime, timedelta

from src.data.loader import fetch_and_prepare_data
from src.optimization.tuner import StrategyTuner
from config.settings import settings


def tune_all():
    print("="*50)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*50)

    full_tuning_results = {}
    
    end_date = datetime.now() - timedelta(days=settings.BACKTEST_TEST_DAYS)
    start_date = end_date - timedelta(days=settings.BACKTEST_HISTORY_BUFFER_DAYS)
    
    strategy_names = ["VolTarget", "Bollinger", "MomentumXGB"]
    
    for ticker in settings.BACKTEST_TICKERS:
        print(f"\n--- Tuning {ticker} ---")
        try:
            df = fetch_and_prepare_data(
                ticker, 
                start_date.strftime("%Y-%m-%d"), 
                end_date.strftime("%Y-%m-%d")
            )
            
            if len(df) < 500:
                print(f"Skipping {ticker}: Not enough data.")
                continue
                
            tuner = StrategyTuner(ticker, df)
            ticker_results = {}
            
            for strat in strategy_names:
                best_params = tuner.optimize(strat, n_trials=10)
                ticker_results[strat] = best_params
                
            full_tuning_results[ticker] = ticker_results
            
            _save_results(full_tuning_results)
        except Exception as e:
            print(f"Error tuning {ticker}: {e}")
            
    print("\n" + "="*50)
    print(f"TUNING COMPLETE. Saved to {settings.TUNED_PARAMS_FILE}")
    print("="*50)


def _save_results(data: dict):
    os.makedirs(os.path.dirname(settings.TUNED_PARAMS_FILE), exist_ok=True)
    with open(settings.TUNED_PARAMS_FILE, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    tune_all()
