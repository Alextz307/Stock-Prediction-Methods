import pandas as pd
import numpy as np
from arch import arch_model
from typing import Any, Literal, cast

from config.settings import settings


class GarchPredictor:
    """
    Wrapper for the ARCH library to handle automated tuning and fitting of GARCH models.
    """

    def __init__(
        self,
        dist: Literal[
            "normal",
            "gaussian",
            "t",
            "studentst",
            "skewstudent",
            "skewt",
            "ged",
            "generalized error",
        ] = "skewt",
    ):
        self.best_p = 1
        self.best_q = 1
        self.dist = dist
        self.model_res: Any = None

    def tune(
        self,
        returns: pd.Series,
        max_p: int = 5,
        max_q: int = 5,
    ) -> tuple[int, int]:
        """
        Performs a Grid Search to find the optimal (p, q) parameters minimizing AIC.
        
        Args:
            returns (pd.Series): Log-returns series.
            max_p (int): Maximum lag order for AR term.
            max_q (int): Maximum lag order for MA term.
            
        Returns:
            tuple[int, int]: Best (p, q).
        """

        best_aic = float("inf")
        best_param = (1, 1)

        scaled_returns = returns * 100.0

        print(f"Tuning GARCH (Max P={max_p}, Max Q={max_q})...")

        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    model = arch_model(
                        scaled_returns,
                        vol="GARCH",
                        p=p,
                        q=q,
                        dist=cast(
                            Literal[
                                "normal",
                                "gaussian",
                                "t",
                                "studentst",
                                "skewstudent",
                                "skewt",
                                "ged",
                                "generalized error",
                            ],
                            self.dist,
                        ),
                        rescale=False,
                    )
                    res = model.fit(disp="off")

                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_param = (p, q)
                except Exception:
                    continue

        self.best_p, self.best_q = best_param
        print(
            f"Optimal GARCH parameters found: p={self.best_p}, q={self.best_q} (AIC: {best_aic:.2f})"
        )
        return best_param


    def fit(self, returns: pd.Series) -> None:
        """
        Fits the GARCH model using the pre-tuned best (p, q).
        Also stores initialization parameters (backcast) to prevent lookahead bias in future generation.
        """
        
        scaled_returns = returns * 100.0
        model = arch_model(
            scaled_returns,
            vol="GARCH",
            p=self.best_p,
            q=self.best_q,
            dist=cast(
                Literal[
                    "normal",
                    "gaussian",
                    "t",
                    "studentst",
                    "skewstudent",
                    "skewt",
                    "ged",
                    "generalized error",
                ],
                self.dist,
            ),
            rescale=False,
        )
        self.model_res = model.fit(disp="off")
        
        if hasattr(self.model_res.model.volatility, "backcast"):
            self.train_backcast = self.model_res.model.volatility.backcast(self.model_res.resid)
        else:
            self.train_backcast = np.var(self.model_res.resid)
            
        self.train_mu = self.model_res.params.get('mu', 0.0)

    def predict_volatility(self, horizon: int = 1) -> float:
        """
        Forecasts future volatility (1-step).
        """
        
        if self.model_res is None:
            raise ValueError("Model must be fit before predicting.")

        forecast = self.model_res.forecast(horizon=horizon)
        var_forecast = forecast.variance.iloc[-1, 0]
        vol_forecast = np.sqrt(var_forecast) / 100.0

        return float(vol_forecast)
    
    def _manual_garch_filter(self, params: pd.Series, residuals: np.ndarray, p: int, q: int, backcast: float) -> np.ndarray:
        """
        Manually filters GARCH(p, q) volatility to ensure strict consistent initialization.
        """

        omega = params['omega']
        alpha = [params.get(f'alpha[{i}]', 0.0) for i in range(1, p + 1)]
        beta = [params.get(f'beta[{i}]', 0.0) for i in range(1, q + 1)]
        
        n = len(residuals)
        sigma2 = np.zeros(n)
        
        for t in range(n):
            val = omega
            
            for i in range(p):
                if t - (i + 1) < 0:
                    val += alpha[i] * backcast
                else:
                    val += alpha[i] * residuals[t - (i + 1)]**2
                    
            for j in range(q):
                if t - (j + 1) < 0:
                    val += beta[j] * backcast
                else:
                    val += beta[j] * sigma2[t - (j + 1)]
            
            sigma2[t] = val
            
        return np.sqrt(sigma2)

    def generate_vol_series(self, returns: pd.Series) -> pd.Series:
        """
        Generates conditional volatility for a new series using FIXED parameters 
        from the training phase, ensuring consistent initialization (NO Lookahead Bias).
        """
        
        if self.model_res is None:
            raise ValueError("Model must be fit first.")
            
        scaled_returns = returns * 100.0
        
        residuals = scaled_returns - self.train_mu
        
        vol_values = self._manual_garch_filter(
            self.model_res.params, 
            residuals.values, 
            self.best_p, 
            self.best_q, 
            self.train_backcast
        )
        
        return pd.Series(vol_values / 100.0, index=returns.index)
