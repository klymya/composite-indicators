from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def plot_hyperplane_2d(
    decision_fn: Callable, x_range: tuple, y_range: tuple, x_step: float, y_step: float,
    colorbar: bool = True, levels=25, cmap: str | Callable = plt.cm.plasma,  alpha: float = 0.8, **kwargs
) -> None:
    """Plot a model's hyperplane contour in 2D.

    Args:
        decision_fn (callable): a model's prediction method: predict, predict_proba, decision_function etc.
        x_range (tuple): (x_min, x_max) - range of x-axis.
        y_range (tuple): (y_min, y_max) - range of y-axis.
        x_step (float): discretization step by x-axis.
        y_step (float): discretization step by x-axis.
        colorbar (bool, optional): add colorbar to the graph. Defaults to True.
        levels (int, optional):  number of the contour lines/regions. Defaults to 25.
        cmap (str | Callable, optional): plot colormap. Defaults to plt.cm.plasma.
        alpha (float, optional): the hyperplane transparency level. Defaults to 0.8.
    """
    xx, yy = np.meshgrid(
        np.arange(*x_range, x_step),
        np.arange(*y_range, y_step)
    )
    X_meshgrid = np.c_[xx.ravel(), yy.ravel()]

    pred_mesh = decision_fn(X_meshgrid)

    plt.contourf(xx, yy, pred_mesh.reshape(xx.shape), levels, cmap=cmap, alpha=alpha, **kwargs)

    if colorbar:
        plt.colorbar()


def plot_hyperplane_3d(
    decision_fn: callable, x_range: tuple, y_range: tuple, x_step: float, y_step: float, fig: Figure = None,
    elev: int = 45, azim: int = 45, colorbar: bool = True, cmap: str | Callable = plt.cm.plasma, alpha: float = 0.8,
    **kwargs
) -> Axes3D:
    """Plot a model's hyperplane in 3D.

    Args:
        decision_fn (callable): a model's prediction method: predict, predict_proba, decision_function and etc.
        x_range (tuple): (x_min, x_max) - range of x-axis.
        y_range (tuple): (y_min, y_max) - range of y-axis.
        x_step (float): discretization step by x-axis.
        y_step (float): discretization step by x-axis.
        fig (Figure, optional): figure to plot surface. Defaults to None.
        elev (int, optional): plot elevation. Defaults to 45.
        azim (int, optional): plot azimut. Defaults to 45.
        colorbar (bool, optional): add colorbar to graph. Defaults to True.
        cmap (str | Callable, optional): plot colormap. Defaults to plt.cm.plasma.
        alpha (float, optional): the hyperplane transparency level. Defaults to 0.8.

    Returns:
        Axes3D: axes with the surface.
    """
    xx, yy = np.meshgrid(
        np.arange(*x_range, x_step),
        np.arange(*y_range, y_step)
    )
    X_meshgrid = np.c_[xx.ravel(), yy.ravel()]
    mesh = decision_fn(X_meshgrid)

    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    serf = ax.plot_trisurf(X_meshgrid[:, 0], X_meshgrid[:, 1], mesh, antialiased=True, cmap=cmap, alpha=alpha, **kwargs)
    if colorbar:
        fig.colorbar(serf, shrink=0.5, aspect=15)
    ax.view_init(elev=elev, azim=azim)

    return ax
