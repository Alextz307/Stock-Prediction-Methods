import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_capital_evolution(
    ticker: str, 
    benchmark_returns: pd.Series, 
    strategy_returns: dict[str, pd.Series], 
    save_dir: str,
    initial_capital: float = 1.0
) -> str:
    """
    Plots the cumulative returns of strategies vs benchmark.
    
    Args:
        ticker (str): Ticker symbol.
        benchmark_returns (pd.Series): Returns of the asset itself.
        strategy_returns (dict[str, pd.Series]): Map of Strategy Name -> Returns Series.
        save_dir (str): Directory to save the chart.
        initial_capital (float): Starting capital for the simulation.
        
    Returns:
        str: Path to the saved image.
    """

    plt.figure(figsize=(10, 6))
    
    cum_bench = (1 + benchmark_returns).cumprod() * initial_capital
    plt.plot(
        cum_bench.index, 
        cum_bench, 
        label="Benchmark (Buy & Hold)", 
        color="black", 
        linestyle="--", 
        alpha=0.6
    )

    for strat_name, returns in strategy_returns.items():
        cum_strat = (1 + returns).cumprod() * initial_capital
        plt.plot(cum_strat.index, cum_strat, label=strat_name)

    plt.title(f"Capital Evolution: {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Capital ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"{ticker}_Strategies.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    
    return save_path
