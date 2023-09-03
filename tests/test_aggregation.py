import numpy as np
import pytest
from unittest.mock import Mock

from composite_indicators.aggregation import _get_centroid, _get_median, _get_medoid, BaseAggregator


def test_get_centroid():
    x = np.array([
        [0, 0],
        [1, 1],
    ])

    centroid = _get_centroid(x)
    assert centroid == pytest.approx([0.5, 0.5], 1e-6)

    x = np.array([
        [0],
        [1],
    ])

    centroid = _get_centroid(x)
    assert centroid == pytest.approx([0.5], 1e-6)


def test_get_median():
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
    ])

    centroid = _get_median(x)
    assert centroid == pytest.approx([0, 1], 1e-6)

    x = np.array([
        [0],
        [1],
        [2]
    ])

    centroid = _get_median(x)
    assert centroid == pytest.approx([1], 1e-6)


def test_get_medoid():
    x = np.array([
        [0, 0],
        [0.4, 0.4],
        [1, 1],
    ])

    centroid = _get_medoid(x)
    assert centroid == pytest.approx([0.4, 0.4], 1e-6)

    x = np.array([
        [0],
        [1],
        [2]
    ])

    centroid = _get_medoid(x)
    assert centroid == pytest.approx([1], 1e-6)


def test_base_aggregator_global_model_none():
    aggregator = BaseAggregator()
    X = np.ones((2, 1))
    y = np.arange(2)

    labels = aggregator._run_global_model(X, y)
    assert (labels == y).all()


def test_base_aggregator_global_model():
    mock = Mock()
    stub_vals = np.ones((2, 1))
    mock.fit_predict = Mock()
    mock.fit_predict.return_value = stub_vals

    aggregator = BaseAggregator(mock, global_alpha=1)
    X = np.ones((2, 1))
    y = np.arange(2)

    labels = aggregator._run_global_model(X, y)
    assert (labels == stub_vals).all()
    mock.fit_predict.assert_called_once()
    assert mock.fit_predict.call_args[0][0].shape[1] == 2


def test_base_aggregator_local_model_none():
    aggregator = BaseAggregator()
    X = np.ones((2, 1))
    y = np.arange(2)

    labels = aggregator._run_local_model(X, y)
    assert (labels == y).all()


def test_base_aggregator_local_model():
    mock = Mock()
    stub_vals = np.ones(1)
    mock.fit_predict = Mock()
    mock.fit_predict.return_value = stub_vals

    aggregator = BaseAggregator(local_model=mock, local_alpha=1)
    X = np.ones((2, 1))
    y = np.arange(2)

    labels = aggregator._run_local_model(X, y)
    assert (labels == np.array([1, 101])).all()
    assert mock.fit_predict.call_count == 2
    assert mock.fit_predict.call_args[0][0].shape[1] == 2


def test_run_aggregation():
    X = np.array([
        [0, 0],
        [1, 1],
        [1, 1]
    ])
    y = np.array([0, 1, 1])
    labels = y.copy()

    aggregator = BaseAggregator()
    X_agg, y_agg = aggregator._run_aggregation(X, y, labels)

    assert X_agg.shape == (2, 2)
    assert y_agg.shape == (2,)
    assert (X_agg == np.array([[0, 0], [1, 1]])).all()
    assert (y_agg == np.array([[0, 1]])).all()


def test_add_regularization():
    x = np.zeros((3, 1))
    y = np.ones(3)

    x_reg = BaseAggregator._add_regularization(x, y, 0)
    assert (x == x_reg).all()

    x_reg = BaseAggregator._add_regularization(x, y, 10)
    assert x_reg.shape[1] == 2
    assert (x_reg[:, :1] == x).all()
    assert (x_reg[:, 1:] == y * 10).all()
