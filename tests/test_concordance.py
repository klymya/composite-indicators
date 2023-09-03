import numpy as np
import pytest

from composite_indicators.concordance import AlphaConcordance, BaseConcordance, GammaSquaredConcordance


@pytest.fixture
def get_data():
    np.random.seed(11)
    x = np.arange(-3.5, 3.5, 0.5)
    w = 2

    y_gt = x * w
    y = y_gt + np.random.randn(len(x)) * 2

    X = x[:, None]

    return X, y, np.array([w]), np.array([-1])


def test_alpha_concordance_w(get_data):
    X, y, w_gt, w_0 = get_data

    _, y_alpha = AlphaConcordance(0).fit_transform(X, y, w=w_0)
    w_alpha = np.linalg.pinv(X) @ y_alpha

    assert w_0 == pytest.approx(w_alpha, 0.1)


def test_alpha_concordance_y(get_data):
    X, y, w_gt, w_0 = get_data

    _, y_alpha = AlphaConcordance(1).fit_transform(X, y, w=w_0)
    assert y == pytest.approx(y_alpha, 0.1)

    _, y_alpha = AlphaConcordance(1, True).fit_transform(X, y, w=w_0)
    assert BaseConcordance.project_y(X, y) == pytest.approx(y_alpha, 0.1)


def test_gamma_squared_concodance_y(get_data):
    X, y, w_gt, w_0 = get_data

    _, y_gamma = GammaSquaredConcordance(0).fit_transform(X, y, w=w_0)
    w_gamma = np.linalg.pinv(X) @ y_gamma

    np.testing.assert_almost_equal(y, y_gamma, 3)
    assert w_gt == pytest.approx(w_gamma, 0.1)

    _, y_gamma = GammaSquaredConcordance(0, True).fit_transform(X, y, w=w_0)
    assert BaseConcordance.project_y(X, y) == pytest.approx(y_gamma, 0.1)


def test_gamma_squared_concodance_w(get_data):
    X, y, w_gt, w_0 = get_data

    _, y_gamma = GammaSquaredConcordance(1e9).fit_transform(X, y, w=w_0)
    w_gamma = np.linalg.pinv(X) @ y_gamma

    assert w_0 == pytest.approx(w_gamma, 0.1)
