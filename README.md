# Advanced Algorithmic Trading & Stock Prediction System

## Abstract

This project implements a sophisticated robust algorithmic trading framework designed to adapt to changing market regimes. By integrating hybrid volatility modeling (GARCH-LSTM), machine learning classifiers (XGBoost), and regime-adaptive technical strategies, the system aims to outperform buy-and-hold benchmarks while significantly reducing drawdown. A key feature of this framework is its automated hyperparameter tuning engine, which utilizes Bayesian optimization (Optuna) to tailor strategy parameters to the unique statistical properties of each asset.

---

## 🚀 Getting Started

This guide explains how to set up the environment and run the experiments locally.

### Prerequisites

*   **Python:** 3.12 or higher
*   **Git:** To clone the repository

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd stock_prediction_methods
    ```

2.  **Set up the environment:**

    **Option A: Using Poetry (Recommended)**
    ```bash
    # Install Poetry if you haven't already
    curl -sSL https://install.python-poetry.org | python3 -

    # Install project dependencies
    poetry install
    ```

    **Option B: Using venv and pip**
    ```bash
    # Create a virtual environment
    python3 -m venv .venv

    # Activate the environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    # .venv\Scripts\activate

    # Install dependencies
    pip install pandas numpy scipy arch torch scikit-learn pydantic pydantic-settings xgboost tabulate yfinance optuna pandas-stubs
    ```

---

## 🧪 Running the Experiments

The project consists of two main stages: **Hyperparameter Tuning** and **Backtesting**.

### 1. Configuration (Optional)
Before running, you can adjust the experiment settings in `config/settings.py`.
*   **`BACKTEST_TICKERS`**: List of stock symbols to analyze (e.g., `["SPY", "AAPL"]`).
*   **`BACKTEST_TEST_DAYS`**: Number of days for the out-of-sample test.
*   **`BACKTEST_HISTORY_BUFFER_DAYS`**: Historical data buffer for training.

### 2. Hyperparameter Tuning
This step uses **Optuna** to find the optimal parameters for each strategy and ticker. It trains on historical data (excluding the test period) and saves the best parameters to `config/tuned_params.json`.

```bash
# If using Poetry
poetry run python -m scripts.tune_strategies

# If using venv/standard python
python -m scripts.tune_strategies
```

*Note: This process may take some time depending on the number of tickers and trials configurations.*

### 3. Backtesting
Once tuning is complete (or if using default parameters), run the backtest to evaluate performance on the out-of-sample test set.

```bash
# If using Poetry
poetry run python -m scripts.run_backtest

# If using venv/standard python
python -m scripts.run_backtest
```

The results (charts, metrics, and logs) will be saved in the `results/` directory.

---

## 1. Introduction

Financial markets are characterized by non-stationary behavior, making static trading strategies prone to failure over long horizons. This project addresses this challenge through three core innovations:
1.  **Hybrid Volatility Forecasting**: Combining econometrics (GARCH) with deep learning (LSTM) to capture both linear and non-linear volatility dynamics.
2.  **Regime-Adaptive Logic**: Strategies that alter their behavior based on market trends (Bull/Bear) and volatility states.
3.  **Automated Parameter Optimization**: A pipeline that rigorously isolates training, validation, and testing data to prevent overfitting while maximizing risk-adjusted returns.

## 2. Methodology

### 2.1 Hybrid Volatility Model
We employ a composite model to forecast realized volatility ($\sigma_{t+1}$):
*   **GARCH(p,q)**: Generalized Autoregressive Conditional Heteroskedasticity models capture volatility clustering (the tendency for high-volatility events to cluster in time).
*   **LSTM Residual Correction**: A Long Short-Term Memory network is trained on the residuals ($\epsilon_t = \sigma_{actual} - \sigma_{GARCH}$) of the GARCH model. It uses market features (RSI, MACD, Returns) to predict the error of the GARCH model.
*   **Final Forecast**: $\hat{\sigma}_{final} = \hat{\sigma}_{GARCH} + \hat{\epsilon}_{LSTM}$.

### 2.2 Trading Strategies

#### A. Volatility Targeting Strategy
*   **Objective**: Maintain a constant annualized risk profile (e.g., Target Volatility = 15%).
*   **Mechanism**: Position sizing is dynamically adjusted.
    $$ Position Size_t = \frac{\text{Target Volatility}}{\text{Forecasted Volatility}_t} $$
*   **Advantage**: Automatically deleverages during turbulent markets (e.g., 2020 crash) and releverages during calm ascents.

#### B. Adaptive Bollinger Breakout
*   **Objective**: Exploit mean-reversion opportunities within the context of the broader trend.
*   **Regime Filter**: Uses a configurable Moving Average (SMA or EMA) to define the regime.
    *   *Bull Market* ($Price > MA_{trend}$): Look for oversold conditions (dip buying).
    *   *Bear Market* ($Price < MA_{trend}$): Look for overbought conditions (short selling).
*   **Dynamic Bands**: Bollinger Band width is set by GARCH-forecasted volatility, not simple standard deviation, making it more responsive to sudden shocks.

#### C. Momentum Gatekeeper (XGBoost)
*   **Objective**: Filter trend-following signals using Machine Learning.
*   **Classifier**: An XGBoost model trained to predict probability of $Return_{t+1} > 0$.
*   **Signal Logic**: Enter Long ONLY if:
    1.  Market is in a structural uptrend ($Price > MA_{slow}$).
    2.  XGBoost probability confidence > Threshold (e.g., 60%).

## 3. Experimental Setup

### 3.1 Data Splitting
To ensure statistical validity and prevent look-ahead bias, we enforce a strict temporal split:
*   **Training/Tuning Period**: 2020-05-18 to 2025-05-09.
    *   Used by Optuna to find optimal parameters (`k`, `window`, etc.).
*   **Out-of-Sample Test Period (Backtest)**: 2025-05-09 to 2026-01-16.
    *   Completely unseen data used solely for final performance evaluation.
    *   *Note: Dates are configurable in settings.*

### 3.2 Hyperparameter Tuning
We used **Optuna** with the Tree-structured Parzen Estimator (TPE) sampler to maximize the **Sharpe Ratio**.
*   **Trials**: Default 10 trials per strategy per ticker.
*   **Parameters Optimized**:
    *   *Bollinger*: Window (10-50), k (1.0-3.0), Trend Window (50-252), MA Type (SMA/EMA).
    *   *VolTarget*: Target Volatility (5%-40%), Trend Window (0-252), MA Type (SMA/EMA).
    *   *XGBoost*: MA Window (10-100), Probability Threshold (51%-70%).
