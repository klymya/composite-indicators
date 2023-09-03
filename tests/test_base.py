from unittest.mock import Mock
import numpy as np

from sklearn.pipeline import Pipeline

from composite_indicators.base import TransformerXYMixin


def test_transformer():
    class A(TransformerXYMixin):
        def _fit_transform(self, X, y, **fit_params):
            pass

    step1 = A()
    step1._fit_transform = Mock()
    step1._fit_transform.return_value = (
        np.ones((10, 1)),
        np.ones(10)
    )

    step2 = A()
    step2._fit_transform = Mock()
    step2._fit_transform.return_value = (
        np.ones((10, 1)) * 2,
        np.ones(10) * 2
    )

    pipe = Pipeline([("step1", step1), ("step2", step2)])

    X = np.zeros((10, 1))
    y = np.zeros(10)
    X_res, y_res = pipe.fit_transform(X, y, step1__w=np.ones_like(y))

    assert (X_res == np.ones((10, 1)) * 2).all()
    assert (y_res == np.ones(10) * 2).all()

    assert (step1._fit_transform.call_args[0][0] == X).all()
    assert (step1._fit_transform.call_args[0][1] == y).all()

    assert (step2._fit_transform.call_args[0][0] == np.ones((10, 1))).all()
    assert (step2._fit_transform.call_args[0][1] == np.ones(10)).all()
