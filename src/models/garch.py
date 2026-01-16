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
        ] = settings.GARCH_DISTRIBUTION,
    ):
        self.best_p = 1
        self.best_q = 1
        self.dist = dist
        self.model_res: Any = None

    def tune(
        self,
        returns: pd.Series,
        max_p: int = settings.GARCH_MAX_P,
        max_q: int = settings.GARCH_MAX_Q,
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

    def generate_vol_series(self, returns: pd.Series) -> pd.Series:
        """
        Generates conditional volatility for a new series using FIXED parameters 
        from the training phase (Valid Out-of-Sample).
        """
        if self.model_res is None:
            raise ValueError("Model must be fit first.")
            
        scaled_returns = returns * 100.0
        
        # Create a new model with the same spec
        from arch import arch_model
        model = arch_model(
            scaled_returns,
            vol="GARCH",
            p=self.best_p,
            q=self.best_q,
            dist=cast(Literal["normal", "t", "skewt"], self.dist),
            rescale=False,
        )
        
        # Fix the parameters to the trained values (no re-training)
        res = model.fix(self.model_res.params)
        
        return res.conditional_volatility / 100.0
