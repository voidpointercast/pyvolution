from typing import Iterator, Sequence
from pyvolution.types.gene import Crossover, Anomaly
from pyvolution.types.population import (
    FitnessFunction, create_children_builder, MateSelector,
    ChildrenSpawn, Population, evaluate_population, top_selector
)
from pyvolution.types.individual import (
    Birthing, create_birth_builder, create_sequential_naming,
    create_gamete_builder, Mitosis, Selector, select_half,
    Individual, Naming
)


def default_mitosis(selector: Selector=select_half, xover=lambda x: x) -> Mitosis:
    return create_gamete_builder(selector, xover)


def default_birth(
        xover: Crossover=lambda x: x,
        anomaly: Anomaly=lambda x: x,
        naming: Naming=create_sequential_naming()
) -> Birthing:
    return create_birth_builder(default_mitosis(), naming, xover=xover, anomaly=anomaly)


def create_fitness_selector(fitness: FitnessFunction, parents: int=2) -> MateSelector:
    def top_breed(population: Population, children: int) -> Iterator[Sequence[Individual]]:
        return top_selector(
            parents,
            tuple(
                individual
                for (individual, ranking) in
                sorted(evaluate_population(fitness, population), key=lambda x: x[1], reverse=True)
            ),
            children
        )
    return top_breed


def top_individuals_breed(
        fitness: FitnessFunction,
        xover: Crossover=lambda x: x,
        anomaly: Anomaly=lambda x: x,
        naming: Naming=create_sequential_naming()
) -> ChildrenSpawn:
    return create_children_builder(create_fitness_selector(fitness), default_birth(xover, anomaly, naming))
