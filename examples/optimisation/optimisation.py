from typing import Sequence
from random import uniform
from math import sin, sqrt
from functools import partial
from pyvolution.models.optimisation.basic import create_basic_model, TestFunction
from pyvolution.evolution import evolve_until, create_step_stop_criteria
from pyvolution.analysis import create_population_subset_store, create_yaml_store, create_yaml_loader
from pyvolution.visualisation.optimisation import create_contour_animation, create_contour_view, show, Normalize

def mutate(x: float) -> float:
    return x + uniform(-0.1, 0.1)


def rosenbrock(xs: Sequence[float]) -> float:
    x, y = xs
    return (1.0 - x)**2 + 100*(y-x**2)**2

def eggholder(xs: Sequence[float]) -> float:
    x, y = xs
    return -(y+47)*sin(sqrt(abs(0.5*x+y+47))-x*sin(sqrt(abs(x-y-47))))



ROSENBROCK = TestFunction(rosenbrock, 2)
SPHERE = TestFunction(lambda xs: sum(x**2 for x in xs), 2)
HIMMELBLAU = TestFunction(lambda xs: (xs[0]**2+xs[1]-11)**2 + (xs[0] + xs[1]**2 -7)**2, 2)
EGGHOLDER = TestFunction(eggholder, 2)


def main():
    #storage = create_population_subset_store(lambda g, i: True, create_yaml_store('results', 'eggholder', False))
    population, evolution, remap = create_basic_model([EGGHOLDER], 100)
    #evolve_until(partial(evolution, mutator=mutate), population, create_step_stop_criteria(500), hooks=[storage])
    loader = create_yaml_loader('results', 'eggholder', False)
    loader = (list(g) for g in loader)
    figure, path = create_contour_view((400, 600, 1.0), (400, 600, 1.0), HIMMELBLAU, levels=1000)
    anim = create_contour_animation(figure, path, 999, loader, remap)
    show()
    anim.save('eggholder.mp4', dpi=200, writer='ffmpeg')


if __name__ == '__main__':
    main()


