from typing import Tuple
import numpy as np

from composite_indicators.base import TransformerXYMixin


class BaseConcordance(TransformerXYMixin):
    """Base class for concordance approaches.
    """
    def __init__(self, enable_projection: bool = False) -> None:
        """Initialize concordance object.

        Args:
            enable_projection (bool, optional): enable y projection onto X subspace. Defaults to False.
        """
        self.enable_projection = enable_projection

    def fit_transform(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the model with X, y and w and apply a concordace to y.

        Args:
            X (np.ndarray): input samples (n_samples, n_features).
            y (np.ndarray): target values (n_samples,).
            w (np.ndarray): esstimated feature weights (n_features,).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                input samples (n_samples, n_features)
                concordant target values (n_samples,)
        """
        if self.enable_projection:
            y_new = self._transform_y(X, self.project_y(X, y), w)
        else:
            y_new = self._transform_y(X, y, w)
        return X, y_new

    @staticmethod
    def project_y(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return X @ np.linalg.pinv(X) @ y

    def _transform_y(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """The method should implemt a concordance algorithm.
        Returns the concordant y.

        Args:
            X (np.ndarray): input samples (n_samples, n_features).
            y (np.ndarray): target values (n_samples,).
            w (np.ndarray): esstimated feature weights (n_features,).

        Returns:
            np.ndarray: concordant target values (n_samples,).
        """
        raise NotImplementedError


class AlphaConcordance(BaseConcordance):
    """Implementation of the alpha concordance approach.
    """
    def __init__(self, alpha: float, enable_projection: bool = False) -> None:
        """Initialize concordance object.

        Args:
            alpha (float): concordance coefficient in [0, 1]
                alpha = 0 - prefer w and ignore y
                alpha = 1 - prefer y and ignore w
            enable_projection (bool, optional): enable y projection onto X subspace. Defaults to False.
        """
        self.alpha = alpha
        super().__init__(enable_projection)

    def _transform_y(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self.alpha * y + (1 - self.alpha) * X @ w


class GammaSquaredConcordance(BaseConcordance):
    """Implementation of the gamma squared concordance approach.
    """
    def __init__(self, gamma_squared: float, enable_projection: bool = False) -> None:
        """Implementation of the gamma squared concordance approach.

        Args:
            gamma_squared (float): concordance coefficient in [0, inf)
                gamma_squared = 0 - prefer y and ignore w
                gamma_squared = inf - prefer w and ignore y
            enable_projection (bool, optional): enable y projection onto X subspace. Defaults to False.
        """
        self.gamma_squared = gamma_squared
        super().__init__(enable_projection)

    def _transform_y(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_pinv = np.linalg.pinv(X)
        X_pinv_T = X_pinv.T

        return np.linalg.inv(np.eye(n_samples) + self.gamma_squared * X_pinv_T @ X_pinv) @ \
            (y + self.gamma_squared * X_pinv_T @ w)
