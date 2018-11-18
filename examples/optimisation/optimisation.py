from typing import Sequence
from random import uniform
from functools import partial
from pyvolution.models.optimisation.basic import create_basic_model, TestFunction
from pyvolution.evolution import evolve_until, create_step_stop_criteria
from pyvolution.analysis import create_population_subset_store, create_yaml_store, create_yaml_loader


def mutate(x: float) -> float:
    return x + uniform(-0.1, 0.1)


def rosenbrock(xs: Sequence[float]) -> float:
    x, y = xs
    return (1.0 - x)**2 + 100*(y-x**2)**2


ROSENBROCK = TestFunction(rosenbrock, 2)



def main():
    storage = create_population_subset_store(lambda g, i: True, create_yaml_store('results', False))
    population, evolution, remap = create_basic_model([ROSENBROCK])
    evolve_until(partial(evolution, mutator=mutate), population, create_step_stop_criteria(1000), hooks=[storage])



if __name__ == '__main__':
    main()


