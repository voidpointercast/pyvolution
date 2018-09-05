from typing import Tuple, Union, Sequence, Callable, Optional, Iterator
from sys import maxsize
from itertools import chain
from random import randint, random
from math import sqrt, isnan
from json import JSONEncoder
from pyvolution.types.gene import create_linear_mapping, create_chromosome_builder, Crossover, remap_genome
from pyvolution.survival import keep_best_halve
from pyvolution.types.population import GrowthDetermination, keep_population_size, Fitness, Survival
from pyvolution.types.individual import create_individual_builder, Naming, create_sequential_naming, Individual
from pyvolution.fitness import create_fitness
from pyvolution.models.algebra import (
    Expression, evaluate, DefaultSeed, DefaultSeedTypes, create_expression_parser, DEFAULT_FUNCTIONS, DEFAULT_VARIABLES,
    create_default_interpretation, Function, create_default_mutator, show_expression
)
from pyvolution.evolution import build_evolution_model
from pyvolution.birth import top_individuals_breed, Birthing
from pyvolution.anomalies import Anomaly


Point = Union[Tuple[float, float], Tuple[float, float, float]]


def default_dominance(genes: Sequence[DefaultSeed]) -> DefaultSeed:
    function_types, values = zip(*genes)
    return DefaultSeedTypes.map_modul(sum(x.value for x in function_types)), sum(values)


def random_seed() -> DefaultSeed:
    return DefaultSeedTypes.map_modul(randint(0, maxsize)), random()


def create_basic_model_fitness(points: Sequence[Point]) -> Fitness:

    def quadratic_error_fitness(expression: Expression) -> float:
        try:
            error = sum(
                (
                    evaluate(
                        expression,
                        dict(zip(DEFAULT_VARIABLES, point[0:-1])),
                        0.0,
                        -float('infinity')
                    ) - point[-1]
                ) ** 2
                for point in points
            )
            if isinstance(error, complex):
                error = abs(error)
            error = -sqrt(error)
            if isnan(error):
                return -float('infinity')
            return error
        except ArithmeticError:
            return -float('infinity')
    return quadratic_error_fitness


def create_basic_model(
        points: Sequence[Point],
        popsize: int=100,
        additional_functions: Sequence[Function]=tuple(),
        chromosome_size: int=64,
        gene_count: int=256,
        karyosize: int=2,
        dominance: Callable[[DefaultSeed], DefaultSeed]=default_dominance,
        xover:Crossover=lambda x: x,
        anomaly: Anomaly=lambda x: x,
        naming: Optional[Naming]=None,
        birth: Optional[Birthing]=None,
        growth: Optional[GrowthDetermination]=None,
        survival: Optional[Survival]=None
):
    """
    :param points:
    :param popsize:
    :param additional_functions:
    :param chromosome_size:
    :param gene_count:
    :param dominance:
    :param xover:
    :param anomaly:
    :param naming:
    :param birth:
    :param growth:
    :param survival:
    :return:
    >>> from itertools import cycle
    >>> points = cycle([[(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (2, 0, 1), (0, 5, 4)]])
    >>> population, evolve, show = create_basic_model(points)
    >>> len(population[0].karyogram)
    4
    >>> len(population[0].karyogram[0])
    2
    >>> mutator = create_default_mutator()
    >>> for i in range(1):
    ...     ranked = tuple(evolve(population, i, 1, mutator))
    ...     population, ranking = zip(*ranked)
    ...     print(max(ranking))
    >>> best = max(ranked, key=lambda x: x[1])[0]
    >>> show_expression(show(best))
    """
    mapping, remapping = create_linear_mapping(chromosome_size)
    parser = create_expression_parser(
        create_default_interpretation(tuple(chain(DEFAULT_FUNCTIONS, additional_functions)))
    )
    ifitness = create_fitness(create_basic_model_fitness(points), remapping, parser, dominance)
    chromosome_builder = create_chromosome_builder(lambda x: x, mapping)
    individual_builder = create_individual_builder(
        chromosome_builder,
        naming if naming else create_sequential_naming(lambda x: x-popsize),
        xover=xover
    )
    evolution = build_evolution_model(
        ifitness,
        birth if birth else top_individuals_breed(ifitness, xover, anomaly),
        growth if growth else keep_population_size(popsize),
        survival if survival else keep_best_halve
    )
    population = tuple(
        individual_builder(((random_seed() for _ in range(gene_count)) for _ in range(karyosize)), 0)
        for _ in range(popsize)
    )
    def to_expression(individual: Individual) -> Expression:
        return remap_genome(remapping, parser, default_dominance, individual.karyogram)

    return population, evolution, to_expression


class BasicAlgebraJSONEncoder(JSONEncoder):
    """
    >>> BasicAlgebraJSONEncoder().encode(Individual([], 0, 'Bob'))
    """

    def __init__(self):
        super().__init__(ensure_ascii=False)

    def default(self, o):
        if isinstance(o, DefaultSeedTypes):
            return o.value

        if isinstance(o, Individual):
            return dict(
                name=o.name,
                generation=o.generation,
                #karyogram=o.karyogram
            )

        return JSONEncoder.default(self, o)