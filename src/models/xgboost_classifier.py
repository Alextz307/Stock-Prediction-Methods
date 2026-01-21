import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import Any

from config.settings import settings


class DirectionalClassifier:
    """
    XGBoost-based classifier for predicting market direction (Up/Down).
    Used as a 'Gatekeeper' to filter trade signals.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        default_params = {
            "objective": "binary:logistic",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 5,
            "random_state": 42,
            "eval_metric": "logloss"
        }
        
        if params:
            default_params.update(params)
            
        self.params = default_params
        self.model = xgb.XGBClassifier(**self.params)

    def set_params(self, **params):
        """
        Updates parameters and re-initializes the model.
        """
        
        self.params.update(params)
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fits the XGBoost model.
        """

        print(f"Training XGBoost Classifier on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns the probability of the Positive Class (Price Increase).
        """

        probs = self.model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=X.index)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns binary predictions (0 or 1).
        """
        
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Evaluates accuracy on a test set.
        """
        
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"XGBoost Accuracy: {acc:.4f}")
        return acc
