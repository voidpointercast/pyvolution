from typing import Sequence, Iterator, Callable, Generator, cast, Iterable, TypeVar, Tuple
from random import choice
from functools import reduce
from operator import add
from itertools import cycle
from pyvolution.types.gene import DataType
from pyvolution.types.individual import Individual, Birthing, Spawning


Fitness = TypeVar('Fitness')
FitnessFunction = Callable[[Individual], Fitness]
Population = Iterable[Individual]
RankedPopulation = Iterable[Tuple[Individual, Fitness]]
MateSelector = Callable[[Population, int], Iterator[Sequence[Individual]]]
ChildrenSpawn = Callable[[Population, int, int], Iterator[Individual]]
EntropySource = Generator[Iterator[DataType], None, None]
Survival = Callable[[RankedPopulation], RankedPopulation]
SurvivalIndication = Callable[[Fitness], bool]
GrowthDetermination = Callable[[RankedPopulation], int]


def build_choice_entropy_source(basis: Sequence[DataType], length: int) -> EntropySource:
    """
    :param basis:
    :param length:
    :return:
    >>> data = [tuple(x) for (i, x) in zip(range(2), build_choice_entropy_source(list(range(4)), 8))]
    >>> len(data[0]) == len(data[1]) == 8
    True
    >>> all(x in range(4) for y in data for x in y)
    True
    """
    while True:
        yield (choice(basis) for _ in range(length))


def build_additive_entropy_source(basis: Sequence[DataType], size: int, length: int) -> EntropySource:
    """
    :param basis:
    :param size:
    :param length:
    :return:
    >>> data = list(tuple(x) for (i, x) in zip(range(3), build_additive_entropy_source('ABCDEFGH', 4, 2)))
    >>> [all(z in 'ABCDEFGH' for y in x for z in y) for x in data]
    [True, True, True]
    """
    while True:
        yield (
            reduce(add, (choice(basis) for _ in range(size-1)), choice(basis))
            for _ in range(length)
        )


def create_population_builder(
        spawn: Spawning,
        entropy: EntropySource
) -> Callable[[None], Generator[Individual, None, None]]:
    """
    :param spawn:
    :param entropy:
    :return:
    >>> from functools import partial
    >>> from string import ascii_uppercase
    >>> from pyvolution.types.gene import create_linear_mapping, create_chromosome_builder
    >>> from pyvolution.types.individual import create_individual_builder, create_sequential_naming
    >>> mapping, remapping = create_linear_mapping(4)
    >>> builder = create_chromosome_builder(list, mapping, handle_gap=lambda x: b'')
    >>> spawner = create_individual_builder(builder, create_sequential_naming())
    >>> pop_builder = create_population_builder(spawner, build_additive_entropy_source(ascii_uppercase, 8, 2))
    >>> population = [x for (i, x) in zip(range(2), pop_builder())]
    >>> list(len(i.karyogram) for i in population)
    [2, 2]
    >>> list(len(cs) for i in population for cs in i.karyogram.values())
    [2, 2, 2, 2]
    >>> list(len(c) for i in population for cs in i.karyogram.values() for c in cs)
    [4, 4, 4, 4, 4, 4, 4, 4]
    """
    def build_population() -> Generator[Individual, None, None]:
        for data in cast(Iterable, entropy):
            yield spawn(data, 0)
    return build_population


def top_selector(num_parents: int, population: Population, amount: int) -> Iterator[Sequence[Individual]]:
    """
    :param num_parents:
    :param population:
    :param amount:
    :return:
    >>> # noinspection PyTypeChecker
    >>> list(top_selector(3, list(range(10)), 4))
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 0, 1)]
    """
    parents = cycle(population)
    return (
        tuple(parent for (i, parent) in zip(range(num_parents), parents))
        for _ in range(amount)
    )


def create_children_builder(selector: MateSelector, birth: Birthing) -> ChildrenSpawn:
    """
    :param selector:
    :param birth:
    :return:
    >>>
    """
    def spawn_children(population: Population, amount: int, generation: int) -> Iterator[Individual]:
        return (
            birth(parents, generation)
            for parents in selector(population, amount)
        )
    return spawn_children


def create_survival_strategy_by_indication(
        fitness_evaluation: Callable[[Fitness], bool]
) -> Survival:
    def determine_survivors(population: RankedPopulation) -> RankedPopulation:
        return (
            (member, fitness) for (member, fitness) in population if fitness_evaluation(fitness)
        )
    return determine_survivors


def evaluate_population(fitness: FitnessFunction, population: Population) -> RankedPopulation:
    return ((member, fitness(member)) for member in population)


def keep_population_size(size: int) -> GrowthDetermination:

    def determine_children_count(population: RankedPopulation) -> int:
        return max(size - len(population), 0)
    return determine_children_count
