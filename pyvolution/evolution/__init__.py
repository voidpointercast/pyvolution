from typing import Iterable
from itertools import chain
from pyvolution.types.individual import Individual
from pyvolution.types.population import (
    ChildrenSpawn, Survival, GrowthDetermination,
    FitnessFunction, evaluate_population, RankedPopulation,
    keep_population_size
)
from pyvolution.survival import keep_best_halve


def build_evolution_model(
        fitness: FitnessFunction,
        birth: ChildrenSpawn,
        growth: GrowthDetermination=keep_population_size(10),
        survival: Survival=keep_best_halve,

):
    """
    :param growth:
    :param birth:
    :param survival:
    :param fitness:
    :return:
    >>> from pyvolution.birth import top_individuals_breed
    >>>
    """

    def evolve(population: Iterable[Individual], generation: int) -> Iterable[RankedPopulation]:
        survivors = survival(evaluate_population(fitness, population))
        return (
            chain(
                evaluate_population(fitness, birth((i for (i, ranking) in survivors), growth(survival), generation+1)),
                survivors
            )
        )

    return evolve
