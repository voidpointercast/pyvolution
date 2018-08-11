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
    >>> from string import ascii_letters
    >>> from random import choice
    >>> from pyvolution.birth import top_individuals_breed
    >>> from pyvolution.fitness import create_fitness
    >>> from pyvolution.types.gene import create_linear_mapping, create_chromosome_builder
    >>> from pyvolution.types.individual import create_individual_builder, create_sequential_naming
    >>> def fitness(value: str) -> float:
    ...     return sum(1 for x in value if x.isupper()) + len(set('HELLO WORLD').intersection(set(value)))
    >>> fitness('HEllo World')
    7
    >>> def randdata(size: int, amount: int):
    ...     return [''.join(choice(ascii_letters) for _ in range(size))  for _ in range(amount)]
    >>> mapping, remapping = create_linear_mapping(4)
    >>> ifitness = create_fitness(fitness, remapping, lambda x: x, lambda x: x[0])
    >>> evolve = build_evolution_model(ifitness, top_individuals_breed(ifitness))
    >>> cbuilder = create_chromosome_builder(lambda x: [x], mapping, lambda x: x)
    >>> builder = create_individual_builder(cbuilder, create_sequential_naming(lambda x: x-10))
    >>> start_pop = [
    ...     builder(randdata(10, 2), 0) for _ in range(10)
    ... ]
    >>> names =  tuple(i.name for i in start_pop)
    >>> min(names), max(names)
    (-10, -1)
    >>> next_gen = tuple(evolve(start_pop, 1))
    >>> len(next_gen)
    10
    >>> names = [i.name for (i, v) in next_gen]
    >>> len([n for n in names if n < 0])
    5
    >>> len([n for n in names if n >= 0])
    5
    """

    def evolve(population: Iterable[Individual], generation: int) -> Iterable[RankedPopulation]:
        survivors = survival(evaluate_population(fitness, population))
        return (
            chain(
                evaluate_population(
                    fitness,
                    birth((i for (i, ranking) in survivors), growth(survivors), generation+1)
                ),
                survivors
            )
        )

    return evolve
