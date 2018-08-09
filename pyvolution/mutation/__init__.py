from typing import Callable
from pyvolution.types.gene import BaseType, Chromosome

Mutator = Callable[[BaseType], BaseType]


def mutate(mutator: Mutator, chromosome: Chromosome) -> Chromosome:
    return type(chromosome)(
        (i, mutator(base)) for (i, base) in chromosome.items()
    )

