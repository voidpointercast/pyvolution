from typing import Iterable
from itertools import chain
from pyvolution.types.individual import Individual
from pyvolution.types.population import (
    ChildrenSpawn, Survival, GrowthDetermination,
    FitnessFunction, evaluate_population, RankedPopulation,
    keep_population_size
)
from pyvolution.survival import keep_best_halve
from pyvolution.mutation import Mutator, mutate


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
    >>> from random import choice, randint
    >>> from collections import defaultdict
    >>> from pyvolution.birth import top_individuals_breed
    >>> from pyvolution.fitness import create_fitness
    >>> from pyvolution.types.gene import create_linear_mapping, create_chromosome_builder
    >>> from pyvolution.types.individual import create_individual_builder, create_sequential_naming
    >>> def fitness(value: str) -> float:
    ...     return sum((1.0 if l.lower() == r.lower() else 0.0) for (l, r) in zip(value, 'HelloWorld'))
    >>> fitness('HElloWorld')
    10.0
    >>> def randdata(size: int, amount: int):
    ...     return [''.join(choice(ascii_letters) for _ in range(size))  for _ in range(amount)]
    >>> mapping, remapping = create_linear_mapping(4)
    >>> ifitness = create_fitness(fitness, remapping, lambda x: x[0])
    >>> evolve = build_evolution_model(ifitness, top_individuals_breed(ifitness))
    >>> cbuilder = create_chromosome_builder(lambda x: [x], mapping)
    >>> builder = create_individual_builder(cbuilder, create_sequential_naming(lambda x: x-10))
    >>> start_pop = [
    ...     builder(randdata(10, 2), 0) for _ in range(10)
    ... ]
    >>> names =  tuple(i.name for i in start_pop)
    >>> min(names), max(names)
    (-10, -1)
    >>> def mutator(value: str) -> str:
    ...     return ''.join(
    ...         ascii_letters[(ascii_letters.index(s) + randint(-1, 1)) % len(ascii_letters)] for s in value
    ...     )
    >>> mutator('B') in ('A', 'B', 'C')
    True
    >>> next_gen = tuple(evolve(start_pop, 1, 1, mutator))
    >>> len(next_gen)
    10
    >>> names = [i.name for (i, v) in next_gen]
    >>> len([n for n in names if n < 0])
    5
    >>> len([n for n in names if n >= 0])
    5
    >>> evaluation = defaultdict(lambda: 0)
    >>> next_gen = start_pop
    >>> i = 0
    >>> while 11.0 not in evaluation and i < 20000:
    ...     next_gen, ranking = zip(*evolve(next_gen, i, 1, mutator))
    ...     evaluation[max(ranking)] += 1
    ...     i += 1
    >>> best = max(evaluate_population(ifitness, next_gen), key=lambda x: x[1])
    >>> best[1] in range(0, 11)
    True
    """

    def evolve(
            population: Iterable[Individual],
            generation: int,
            steps: int=1,
            mutator: Mutator=lambda x: x
    ) -> Iterable[RankedPopulation]:
        if not steps:
            return evaluate_population(fitness, population)

        survivors = survival(evaluate_population(fitness, population))
        next_gen = (
            chain(
                evaluate_population(
                    fitness,
                    (
                        mutate(mutator, child)
                        for child in birth((i for (i, ranking) in survivors), growth(survivors), generation+1)
                    )
                ),
                survivors
            )
        )
        return evolve(
            (
                i for (i, ranking) in next_gen
             ),
            generation+1,
            steps-1,
            mutator
        )

    return evolve
