from typing import Callable, Sequence, Optional, List
from math import sqrt, isnan
from random import uniform
from pyvolution.naming import create_default_naming
from pyvolution.evolution import build_evolution_model
from pyvolution.types.individual import Individual, Naming, create_individual_builder, Spawning
from pyvolution.fitness import FitnessFunction, create_fitness
from pyvolution.types.gene import Crossover, Anomaly, create_linear_mapping, Transcription, ReverseTranscription, create_chromosome_builder
from pyvolution.types.population import Birthing, GrowthDetermination, Survival, keep_population_size
from pyvolution.birth import top_individuals_breed
from pyvolution.survival import keep_best_halve
from pyvolution.models.optimisation import TestFunction



def create_model_fitness(funcs: Sequence[TestFunction]):
    def optimisation_function_fitness(values: Sequence[Sequence[float]]) -> float:
        error = -sqrt(abs(sum(f(*arg)**2 for (f, arg) in zip(funcs, values))))
        return error if not isnan(error) else -float('infinity')
    return optimisation_function_fitness


def create_random_arguments_creator(a: float, b: float) -> Callable[[int], Sequence[float]]:
    def create_random_arguments(size: int) -> Sequence[float]:
        return [uniform(a, b) for _ in range(size)]

    return create_random_arguments

def create_basic_model(
        test_functions: Sequence[TestFunction],
        popsize: int=100,
        karyosize: int=2,
        dominance: Callable[[Sequence[float]], float]=sum,
        xover: Crossover = lambda x: x,
        anomaly: Anomaly = lambda x: x,
        naming: Optional[Naming] = None,
        birth: Optional[Birthing] = None,
        growth: Optional[GrowthDetermination] = None,
        survival: Optional[Survival] = None,
        random: Callable[[int], Sequence[float]]=create_random_arguments_creator(-10, 10)
):
    """
    :param test_functions:
    :param popsize:
    :param karyosize:
    :param dominance:
    :param xover:
    :param anomaly:
    :param naming:
    :param birth:
    :param growth:
    :param survival:
    :param random:
    :return:
    >>> population, evolution, fitness = create_basic_model([TestFunction(lambda xs: sum(map(abs, xs)))])
    >>> sample = population[0]
    >>> print(sample)
    >>> fitness(sample)
    """
    arities = set(f.arity for f in test_functions)
    if len(arities) != 1:
        raise AttributeError("Basic optimisation model requires functions of same arity.")
    arity: int = arities.pop()
    chromosome_size: int = arity
    gene_count: int = len(test_functions)
    def transcription(args: Sequence[Sequence[float]]) -> Sequence[float]:
        return [value for arg in args for value in arg]

    def reverse_transcription(genes: Sequence[float]) -> Sequence[Sequence[float]]:
        args: List[List[float]] = list()
        for (i, gene) in enumerate(genes):
            if i % arity == 0:
                args.append([])
            args[-1].append(gene)
        return args


    mapping, remapping = create_linear_mapping(chromosome_size)
    fitness = create_model_fitness(test_functions)
    ifitness = create_fitness(fitness, remapping, reverse_transcription, dominance)
    chromosome_builder = create_chromosome_builder(transcription, mapping)
    naming = naming if naming else create_default_naming()
    individual_builder: Spawning = create_individual_builder(
        chromosome_builder,
        naming,
        xover=xover
    )
    evolution = build_evolution_model(
        ifitness,
        birth if birth else top_individuals_breed(ifitness, xover, anomaly, naming),
        growth if growth else keep_population_size(popsize),
        survival if survival else keep_best_halve
    )

    population = [
        individual_builder([[random(arity) for _ in range(gene_count)] for _ in range(karyosize)], 0)
    ]
    return population, evolution, ifitness






