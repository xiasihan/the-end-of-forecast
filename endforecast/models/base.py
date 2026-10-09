"""
Abstract base class for all prediction models.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
import pandas as pd


class BaseModel(ABC):
    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        self.params = params or {}
        self._model: Any = None
        self._fitted = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> "BaseModel":
        self._model = self._fit_impl(X, y, **kwargs)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict().")
        return self._predict_impl(X, **kwargs)

    def cross_validate(self, X: pd.DataFrame, y: np.ndarray,
                       n_splits: int = 5, **kwargs: Any) -> dict[str, Any]:
        from sklearn.model_selection import TimeSeriesSplit, KFold
        splitter = TimeSeriesSplit(n_splits=n_splits) if kwargs.get("cv_strategy") == "rolling_window" else KFold(n_splits=n_splits, shuffle=False)
        scores = []
        for train_idx, val_idx in splitter.split(X):
            self.fit(X.iloc[train_idx], y[train_idx])
            pred = self.predict(X.iloc[val_idx])
            scores.append(float(np.mean((pred - y[val_idx]) ** 2)))
        return {"cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores)), "cv_scores": scores}

    @abstractmethod
    def _fit_impl(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> Any: ...

    @abstractmethod
    def _predict_impl(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray: ...