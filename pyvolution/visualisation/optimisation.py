from typing import Tuple, Callable, Sequence, Optional, Iterable, Any
from itertools import product
from numpy import arange, meshgrid, ndarray, sqrt, c_
from matplotlib.pyplot import figure, contourf, show, scatter, Figure, Normalize
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation
from matplotlib.cm import get_cmap
from pyvolution.models.optimisation import TestFunction
from pyvolution.types.population import Population
from pyvolution.types.individual import Individual



def create_contour_view(
        xrange: Tuple[float, float, float],
        yrange: Tuple[float, float, float],
        target: TestFunction,
        levels: Optional[int]=None,
        colormap: Optional[str]='jet',
        normalisation: Optional[Normalize]=None

) -> Tuple[Figure, PatchCollection]:
    """
    :param xrange:
    :param yrange:
    :param target:
    :return:
    """


    xs = arange(*xrange)
    ys = arange(*yrange)

    values = []
    for x in xs:
        values.append([])
        for y in ys:
            values[-1].append(target(x, y))


    fig = figure()
    contourf(xs, ys, values, levels=levels, cmap=get_cmap(colormap), norm=normalisation)
    return fig, scatter([], [], s=5, color='red', marker='o')


def create_contour_animation(
        contour: Figure,
        scatter_data: PatchCollection,
        frames: int,
        generations: Iterable[Population],
        remapping: Callable[[Individual], Sequence[float]],
        interval: int=200
):
    """
    :param contour:
    :param frames:
    :param generations:
    :param remapping:
    :return:
    >>> test = TestFunction(lambda xs: sum(x**2 for x in xs), 2)
    >>> fig, sc = create_contour_view((-2, 2, 0.1), (-2, 2, 0.1), test, 20)
    >>> type(sc)
    >>> anim = create_contour_animation(fig, sc, 2, iter([[(-1, -1), (1, 1)], [(-1, 0), (0.5, 0.5)]]), lambda x: x)
    >>> show()
    >>> anim.save('line.gif', dpi=80, writer='imagemagick')
    """
    points = (
        [remapping(i) for i in gen]
        for gen in generations
    )
    def update(step: int):
        pxs, pys = tuple(zip(*next(points)))
        scatter_data.set_offsets(c_[pxs, pys])
        return scatter_data, contour.axes[0]

    anim = FuncAnimation(contour, update, frames=frames, interval=interval)
    return anim

