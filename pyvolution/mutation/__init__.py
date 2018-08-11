from typing import Callable
from pyvolution.types.gene import BaseType
from pyvolution.types.individual import Individual

Mutator = Callable[[BaseType], BaseType]


def mutate(mutator: Mutator, individual: Individual) -> Individual:
    """
    :param mutator:
    :param individual:
    :return:
    >>> member = Individual({0: [{0: 1, 1: 2}], 1: ({0: 3, 1:4},)}, 0, 'John doe')
    >>> mutate(lambda x: x**2, member)
    Individual(karyogram={0: [{0: 1, 1: 4}], 1: ({0: 9, 1: 16},)}, generation=0, name='John doe')
    """

    mutated = (
        (
            position,
            type(chromosomes)(
                type(chromosome)(
                    (p, mutator(base)) for (p, base) in chromosome.items()
                )
                for chromosome in chromosomes
            )
        )
        for (position, chromosomes) in individual.karyogram.items()
    )
    mutant = Individual(**vars(individual))
    mutant.karyogram = type(individual.karyogram)(mutated)
    return mutant

