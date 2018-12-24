from typing import Callable, Iterable, Dict, Tuple, Sequence, Optional
from matplotlib.pyplot import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from pyvolution.types.individual import Individual
from pyvolution.models.algebra import Expression, evaluate, DomainType, create_function_from_expression


def create_function_visualisation(
        variables: Sequence[str],
        remapping: Callable[[Individual], Expression],
        zero: DomainType=0.0,
        default: Optional[DomainType]=None
):
    def create_animation(
            base: Figure,
            arguments: Sequence[Sequence[DomainType]],
            individuals: Iterable[Individual],
            **kwargs

    ):
        def update(step: int):
            func = create_function_from_expression(remapping(next(individuals)), variables, zero, default)
            points = (
                list(arg) + [func(*arg)]
                for arg in arguments
            )
        anim = FuncAnimation(base, update, **kwargs)
        return anim

    return create_animation
