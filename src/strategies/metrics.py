import pandas as pd
import numpy as np


def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    """
    Computes performance metrics for a series of returns.
    
    Args:
        returns (pd.Series): Daily percentage returns.
        risk_free_rate (float): Annualized risk-free rate (decimal).
        
    Returns:
        dict: {
            "Total Return": Absolute return over the period,
            "Annualized Return": Mean annual return,
            "Annualized Volatility": Annual standard deviation,
            "Sharpe Ratio": Risk-adjusted return,
            "Sortino Ratio": Downside risk-adjusted return,
            "Max Drawdown": Maximum peak-to-trough decline,
            "Win Rate": Percentage of positive days
        }
    """
    
    if len(returns) < 2:
        return {}

    ann_factor = 252

    total_return = (1 + returns).prod() - 1

    mean_return = returns.mean() * ann_factor

    volatility = returns.std() * np.sqrt(ann_factor)

    sharpe = 0.0
    if volatility > 1e-6:
        sharpe = (mean_return - risk_free_rate) / volatility

    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(ann_factor)
    
    sortino = 0.0
    if downside_vol > 1e-6:
        sortino = (mean_return - risk_free_rate) / downside_vol

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    wins = len(returns[returns > 0])
    total = len(returns)
    win_rate = wins / total if total > 0 else 0.0

    return {
        "Total Return": total_return,
        "Annualized Return": mean_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }
