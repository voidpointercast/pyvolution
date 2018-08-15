from typing import Callable
from pyvolution.types.individual import Individual
from pyvolution.types.gene import remap_genome, GeneRemapping, GeneDecoding, ReverseTranscription, Dominance
from pyvolution.types.population import FitnessFunction, Fitness, DataType


def create_fitness(
        func: Callable[[DataType], Fitness],
        remapping: GeneRemapping,
        retranscribe: ReverseTranscription,
        dominance: Dominance=lambda x: max(filter(None, x))
) -> FitnessFunction:
    def fitness(individual: Individual) -> Fitness:
        return func(remap_genome(remapping, retranscribe, dominance, individual.karyogram))
    return fitness






