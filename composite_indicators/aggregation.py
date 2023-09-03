from typing import Tuple
from enum import Enum

import numpy as np
from sklearn.base import ClusterMixin
from scipy.spatial import distance_matrix

from composite_indicators.base import TransformerXYMixin


def _get_centroid(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=0)


def _get_median(x: np.ndarray) -> np.ndarray:
    return np.median(x, axis=0)


def _get_medoid(x: np.ndarray) -> np.ndarray:
    return x[np.argmin(distance_matrix(x, x).sum(axis=0))]


class Aggregation(Enum):
    """Types of available aggregation methods.
    """
    CENTROID = _get_centroid
    MEDOID = _get_medoid
    MEDIAN = _get_median


class BaseAggregator(TransformerXYMixin):
    """
    The class implements an aggregation approach to dataset size reduction.
    It uses two stages:
    1. Global - group data into big clusters
    2. Local - splits each big cluster into subclusters and applies aggregation method.

    Also, clusterization uses regularization based on target value:
    additional dimension with target values multiplied by the coefficient.
    This allows for taking into account labels during distance calculation.
    """
    def __init__(
        self, global_model: ClusterMixin = None, global_alpha: float = 0,
        local_model: ClusterMixin = None, local_alpha: float = 0, agg: Aggregation = Aggregation.CENTROID
    ) -> None:
        """Initialize aggregation model.

        Args:
            global_model (ClusterMixin, optional): clusterization model for global step. Defaults to None.
            global_alpha (float, optional): regularization coefficient for global step >= 0. Defaults to 0.
            local_model (ClusterMixin, optional): clusterization model for local step. Defaults to None.
            local_alpha (float, optional): regularization coefficient for local step >= 0. Defaults to 0.
            agg (Aggregation, optional): aggregation method: centroid, median or medoid.
                Defaults to Aggregation.CENTROID.
        """
        assert global_alpha >= 0, "`global_alpha` should be >= 0."
        assert local_alpha >= 0, "`local_alpha` should be >= 0."

        self.global_model = global_model
        self.global_alpha = global_alpha
        self.local_model = local_model
        self.local_alpha = local_alpha
        self.agg = agg

    @staticmethod
    def _add_regularization(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        return np.hstack([X, alpha * y[:, None]]) if alpha else X

    def _run_global_model(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.global_model:
            labels = self.global_model.fit_predict(self._add_regularization(X, y, self.global_alpha))
        else:
            labels = y.copy()

        return labels

    def _run_local_model(self, X, labels):
        if self.local_model:
            labels_local = np.zeros_like(labels)
            X_in = self._add_regularization(X, labels, self.local_alpha)
            for idx, ulabel in enumerate(np.unique(labels)):
                labels_local[labels == ulabel] = \
                    self.local_model.fit_predict(X_in[labels == ulabel]) + idx * 100

            labels = labels_local

        return labels

    def _run_aggregation(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray) -> np.ndarray:
        X_agg = []
        y_agg = []
        for ulabel in np.unique(labels):
            X_agg.append(self.agg(X[labels == ulabel]))
            y_cl = y[labels == ulabel]
            y_tmp = y_cl[0] if self.global_model is None else self.agg(y_cl)
            y_agg.append(y_tmp)

        X_agg = np.vstack(X_agg)

        return X_agg, np.array(y_agg)

    def _fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the model with X, y and apply an aggreagation to the data.

        Args:
            X (np.ndarray): input samples (n_samples, n_features).
            y (np.ndarray): target values (n_samples,).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                aggregated samples (n_samples_agg, n_features), n_samples_agg <= n_samples
                aggregated target values (n_samples_agg,)
        """
        labels = self._run_global_model(X, y)
        labels = self._run_local_model(X, labels)

        return self._run_aggregation(X, y, labels)


class ClassificationAggregator(BaseAggregator):
    def __init__(
        self, local_model: ClusterMixin, local_alpha: float = 0, agg: Aggregation = Aggregation.CENTROID, **kwargs
    ) -> None:
        """Initialize aggregation model for classification data.

        Args:
            local_model (ClusterMixin, optional): clusterization model for local step. Defaults to None.
            local_alpha (float, optional): regularization coefficient for local step >= 0. Defaults to 0.
            agg (Aggregation, optional): aggregation method: centroid, median or medoid.
                Defaults to Aggregation.CENTROID.
        """
        super().__init__(local_model=local_model, local_alpha=local_alpha, agg=agg)


class RegressionAggregator(BaseAggregator):
    def __init__(
        self, global_model: ClusterMixin = None, global_alpha: float = 0,
        local_model: ClusterMixin = None, local_alpha: float = 0, agg: Aggregation = Aggregation.CENTROID
    ) -> None:
        """Initialize aggregation model for regression data.

        Args:
            global_model (ClusterMixin, optional): clusterization model for global step. Defaults to None.
            global_alpha (float, optional): regularization coefficient for global step >= 0. Defaults to 0.
            local_model (ClusterMixin, optional): clusterization model for local step. Defaults to None.
            local_alpha (float, optional): regularization coefficient for local step >= 0. Defaults to 0.
            agg (Aggregation, optional): aggregation method: centroid, median or medoid.
                Defaults to Aggregation.CENTROID.
        """
        super().__init__(
            global_model=global_model,
            global_alpha=global_alpha,
            local_model=local_model,
            local_alpha=local_alpha,
            agg=agg
        )
