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
