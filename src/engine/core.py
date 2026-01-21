import pandas as pd
import os
import json
import copy
from datetime import datetime, timedelta

from src.data.loader import fetch_and_prepare_data
from src.strategies.strategies import BaseStrategy
from src.strategies.metrics import calculate_metrics
from src.visualization.plots import plot_capital_evolution
from config.settings import settings


class BacktestEngine:
    """
    Orchestrates the backtesting simulation.

    Responsibilities:
        - Fetching data for multiple tickers.
        - Splitting Train/Test sets.
        - Executing strategies iteratively.
        - Calculating performance metrics (Returns, Sharpe, Drawdown).
        - Generating visualizations and reports.
    """

    def __init__(
        self,
        strategies: dict[str, BaseStrategy],
        tickers: list[str],
        test_days: int = 252,
        results_dir: str = "results",
    ):
        """
        Args:
            strategies (dict): Name -> Strategy Object map.
            tickers (list): List of ticker symbols to test.
            test_days (int): Number of days to include in the out-of-sample test.
            results_dir (str): Directory for output.
        """

        self.strategies = strategies
        self.tickers = tickers
        self.test_days = test_days
        self.results_dir = results_dir
        self.charts_dir = os.path.join(results_dir, "charts")

        self.tuned_params = {}
        if os.path.exists(settings.TUNED_PARAMS_FILE):
             with open(settings.TUNED_PARAMS_FILE, "r") as f:
                 self.tuned_params = json.load(f)
             print(f"Loaded tuned parameters from {settings.TUNED_PARAMS_FILE}")

        os.makedirs(self.charts_dir, exist_ok=True)

        self.all_results: list[dict] = []

    def run(self) -> None:
        print("\n" + "=" * 50)
        print(f"STARTING COMPREHENSIVE BACKTEST")
        print("=" * 50 + "\n")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=5 * 365)

        for ticker in self.tickers:
            try:
                self._process_ticker(ticker, start_date, end_date)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                import traceback

                traceback.print_exc()

        self._finalize()

    def _process_ticker(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> None:
        print(f"\nProcessing {ticker}...")

        df = fetch_and_prepare_data(
            ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        if len(df) <= self.test_days:
            print(f"Skipping {ticker}: Not enough data.")
            return

        train_df = df.iloc[: -self.test_days]
        test_df = df.iloc[-self.test_days :]

        if len(train_df) < 500:
            print(f"Skipping {ticker}: Training set too small.")
            return

        asset_returns = test_df["close"].pct_change().dropna()
        self._record_result(ticker, "Benchmark", asset_returns)

        ticker_strat_returns = {}

        for name, default_strategy in self.strategies.items():
            print(f"  > Running {name}...")

            strategy = default_strategy
            if ticker in self.tuned_params and name in self.tuned_params[ticker]:
                params = self.tuned_params[ticker][name]
                print(f"    [Optimized] Using params: {params}")
                strategy = default_strategy.__class__(**params)
            else:
                strategy = copy.deepcopy(default_strategy)

            if hasattr(strategy, "train"):
                strategy.train(train_df)

            full_signals = strategy.generate_signals(df)

            test_signals = full_signals.loc[test_df.index]

            aligned_signals = test_signals.shift(1).dropna()

            common_idx = aligned_signals.index.intersection(asset_returns.index)
            aligned_signals = aligned_signals.loc[common_idx]
            actual_returns = asset_returns.loc[common_idx]

            raw_strat_return = aligned_signals * actual_returns

            turnover = aligned_signals.diff().abs().fillna(0)
            costs = turnover * settings.BACKTEST_TRANSACTION_FEE

            net_returns = raw_strat_return - costs

            ticker_strat_returns[name] = net_returns
            self._record_result(ticker, name, net_returns)

        chart_path = plot_capital_evolution(
            ticker, 
            asset_returns, 
            ticker_strat_returns, 
            self.charts_dir,
            initial_capital=settings.INITIAL_CAPITAL
        )
        print(f"  > Saved chart: {chart_path}")

    def _record_result(self, ticker: str, strat_name: str, returns: pd.Series) -> None:
        metrics = calculate_metrics(returns)
        metrics["Ticker"] = ticker
        metrics["Strategy"] = strat_name
        self.all_results.append(metrics)

        print(
            f"    {strat_name} | Sharpe: {metrics.get('Sharpe Ratio', 0):.2f} | Return: {metrics.get('Total Return', 0)*100:.2f}%"
        )

    def _finalize(self) -> None:
        if not self.all_results:
            print("No results generated.")
            return

        res_df = pd.DataFrame(self.all_results)
        cols = [
            "Ticker",
            "Strategy",
            "Sharpe Ratio",
            "Total Return",
            "Max Drawdown",
            "Win Rate",
        ]
        cols = [c for c in cols if c in res_df.columns]

        res_df = res_df[cols]

        print("\n" + "=" * 50)
        print("FINAL RESULTS")
        print("=" * 50)
        print(res_df.to_string(index=False))
