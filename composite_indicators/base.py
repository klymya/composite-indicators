from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator


class TransformerXYMixin(BaseEstimator, ABC):
    """Mixin class for all transformers in composite-indicators.

    The interface for transforming the `X` and `y` data.
    """
    @staticmethod
    def _get_xy(X: tuple | np.ndarray, y: np.ndarray) -> np.ndarray:
        if isinstance(X, np.ndarray):
            return X, y
        else:
            return X

    @staticmethod
    def _get_x(X: tuple | np.ndarray) -> np.ndarray:
        if isinstance(X, np.ndarray):
            return X
        else:
            return X[0]

    @abstractmethod
    def _fit_transform(self, X: np.ndarray, y: np.ndarray, **fit_params) -> Tuple[np.ndarray, np.ndarray]:
        """Fit to data, then transform it. The method should implements the transformation logic.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X` and `y`.

        Args:
            X (np.ndarray): input samples (n_samples, n_features).
            y (np.ndarray): target values (n_samples,).

        Returns:
            Tuple[np.ndarray, np.ndarray]: transformed version of `X` and `y`:
                X_new (np.ndarray): transformed samples (n_samples_new, n_features_new)
                y_new (np.ndarray): transformed target values (n_samples_new,)
        """

    def fit_transform(self, X: np.ndarray, y: np.ndarray, **fit_params) -> Tuple[np.ndarray, np.ndarray]:
        """Fit to data, then transform it. Public interface.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X` and `y`.

        Args:
            X (np.ndarray): input samples (n_samples, n_features).
            y (np.ndarray): target values (n_samples,).

        Returns:
            Tuple[np.ndarray, np.ndarray]: transformed version of `X` and `y`:
                X_new (np.ndarray): transformed samples (n_samples_new, n_features_new)
                y_new (np.ndarray): transformed target values (n_samples_new,)
        """
        return self._fit_transform(*self._get_xy(X, y), **fit_params)

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self._get_x(X)

    def _more_tags(self):
        return {"requires_y": True}
