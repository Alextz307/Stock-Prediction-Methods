import json
import os
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.data.loader import fetch_and_prepare_data
from src.optimization.tuner import StrategyTuner
from config.settings import settings


def tune_all() -> None:
    print("="*50)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*50)


def tune_ticker(ticker: str) -> tuple[str, dict] | None:
    """
    Tunes a single ticker and returns (ticker, results_dict).
    """

    end_date = datetime.now() - timedelta(days=settings.BACKTEST_TEST_DAYS)
    start_date = end_date - timedelta(days=settings.BACKTEST_HISTORY_BUFFER_DAYS)
    
    strategy_names = ["VolTarget", "Bollinger", "MomentumXGB"]
    
    print(f"\n--- Tuning {ticker} (Process ID: {os.getpid()}) ---")
    try:
        df = fetch_and_prepare_data(
            ticker, 
            start_date.strftime("%Y-%m-%d"), 
            end_date.strftime("%Y-%m-%d")
        )
        
        if len(df) < 500:
            print(f"Skipping {ticker}: Not enough data.")
            return None
            
        tuner = StrategyTuner(ticker, df)
        ticker_results = {}
        
        for strat in strategy_names:
            best_params = tuner.optimize(strat, n_trials=settings.TUNING_N_TRIALS)
            ticker_results[strat] = best_params
            
        return ticker, ticker_results
    except Exception as e:
        print(f"Error tuning {ticker}: {e}")
        return None


def tune_all() -> None:
    print("="*50)
    print("STARTING PARALLEL HYPERPARAMETER TUNING")
    print("="*50)

    full_tuning_results = {}

    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = {executor.submit(tune_ticker, ticker): ticker for ticker in settings.BACKTEST_TICKERS}
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                if result:
                    res_ticker, res_data = result
                    full_tuning_results[res_ticker] = res_data
                    _save_results(full_tuning_results)
            except Exception as e:
                print(f"Wrapper exception for {ticker}: {e}")

    print("\n" + "="*50)
    print(f"TUNING COMPLETE. Saved to {settings.TUNED_PARAMS_FILE}")
    print("="*50)


def _save_results(data: dict) -> None:
    os.makedirs(os.path.dirname(settings.TUNED_PARAMS_FILE), exist_ok=True)
    with open(settings.TUNED_PARAMS_FILE, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    tune_all()
