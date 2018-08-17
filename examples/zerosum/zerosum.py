from typing import Sequence
from random import randint
from itertools import chain
from pyvolution.types.gene import create_linear_mapping, create_chromosome_builder, remap_genome
from pyvolution.anomalies import create_chromosomial_anomaly
from pyvolution.fitness import create_fitness
from pyvolution.evolution import build_evolution_model
from pyvolution.birth import top_individuals_breed
from pyvolution.types.population import keep_population_size
from pyvolution.types.individual import create_individual_builder, create_sequential_naming


def create_random_data(length: int, amount: int) -> Sequence[Sequence[int]]:
    return tuple(
        tuple(randint(-10, 10) for _ in range(length))
        for _ in range(amount)
    )

def mutate(value: int) -> int:
    return value + randint(-2, 2)


def resize_anomaly(values: Sequence[int]) -> Sequence[int]:
    if randint(-10, 1) >= 0:
        return tuple(chain(values, [randint(-10, 10)]))
    else:
        return values


def zerosum(popsize: int, cycles: int):
    """
    :param popsize:
    :param cycles:
    :return:
    >>> zerosum(1000, 1000)
    """
    def fitness(values: Sequence[int]) -> float:
        return -abs(sum(values)) - abs(max(values)) -abs(min(values))

    mapping, remapping = create_linear_mapping(5)
    anomaly = create_chromosomial_anomaly(lambda x: x, lambda x: x, resize_anomaly)
    cbuilder = create_chromosome_builder(lambda x: x, mapping)
    builder = create_individual_builder(cbuilder, create_sequential_naming(lambda x: x-popsize))
    ifitness = create_fitness(fitness, remapping, lambda x: x, sum)
    evolve = build_evolution_model(
        ifitness,
        top_individuals_breed(ifitness, anomaly=anomaly),
        keep_population_size(100)
    )
    population = [
        (builder(create_random_data(10, 2), 0), 0) for _ in range(popsize)
    ]
    for cycle in range(cycles):
        population = evolve((i for (i, score) in population), cycle, 1, mutate)
    best = max(population, key=lambda x: x[1])
    print("Best individual with data={0} and score {1}".format(
        remap_genome(remapping, lambda x: x, sum, best[0].karyogram),
        best[1]
    )
    )

