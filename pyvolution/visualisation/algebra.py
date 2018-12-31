from typing import Callable, Iterable, Dict, Tuple, Sequence, Optional
from itertools import cycle
from matplotlib.pyplot import Figure, Axes, subplots
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from numpy import arange, nan
from pyvolution.types.individual import Individual
from pyvolution.models.algebra import Expression, evaluate, DomainType, create_function_from_expression


def create_1d_function_visualisation(
        variables: Sequence[str],
        remapping: Callable[[Individual], Expression],
        zero: DomainType=0.0,
        default: Optional[DomainType]=None,
        yrange: Tuple[float, float]=(-10.0, 10.0)
):
    """
    :param variables:
    :param remapping:
    :param zero:
    :param default:
    :return:
    >>> from pyvolution.models.algebra import DefaultSeedTypes, create_expression_parser
    >>> from itertools import cycle
    >>> parser = create_expression_parser()
    >>> seed = [
    ...     (DefaultSeedTypes.FUNCTION, 0.0),
    ...     (DefaultSeedTypes.CONSTANT, 1.0),
    ...     (DefaultSeedTypes.VARIABLE, 0.0)
    ... ]
    >>> expr = parser(seed)
    >>> create_animation = create_1d_function_visualisation(['x'], lambda x: x)
    >>> fig, ax = subplots()
    >>> fig.suptitle('{0}')
    >>> anim = create_animation(fig, ax, arange(-2,2,0.1), iter([expr]*100), frames=99, interval=200, blit=False)
    >>> ax.legend()
    >>> anim.save('algebra.gif', dpi=80, writer='imagemagick')
    """
    def create_animation(
            figure: Figure,
            ax: Axes,
            arguments: Sequence[DomainType],
            individuals: Iterable[Individual],
            title: str='{0}',
            name: str='Approximation',
            **kwargs

    ):
        x_min, x_max = min(arguments), max(arguments)
        xs = arange(x_min, x_max, (x_max - x_min)/1000.0)
        line, = ax.plot(xs, [0]*len(xs), label=name)
        text = figure._suptitle.get_text()

        def init():
            line.set_ydata([yrange[0], yrange[1]]*(len(xs) // 2))
            return line,

        def update(step: int):
            figure._suptitle.set_text(title.format(step))
            func = create_function_from_expression(remapping(next(individuals)), variables, zero, default)
            values = tuple(
                func(x)
                for x in xs
            )
            line.set_ydata(values)
            return line,

        anim = FuncAnimation(figure, update, init_func=init, **kwargs)
        return anim

    return create_animation
