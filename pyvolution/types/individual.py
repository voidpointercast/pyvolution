from typing import TypeVar, Callable, Sequence, Iterator, Optional, Tuple
from random import choice
from itertools import groupby, count
from attr import attrs, attrib
from pyvolution.types.gene import DataType, Chromosome, KaryoTranscription, Karyogram, Crossover

NameType = TypeVar('NameType')


@attrs
class Individual:
    karyogram: Karyogram = attrib()
    generation: int = attrib()
    name: NameType = attrib(default=None)


Spawning = Callable[[Iterator[DataType], int], Individual]
Naming = Callable[[int, Sequence[Individual]], NameType]
Selector = Callable[[Karyogram], Karyogram]
Gamete = Karyogram
Mitosis = Callable[[Individual], Gamete]
Birthing = Callable[[Sequence[Individual], int], Individual]
Merging = Callable[[Iterator[Karyogram]], Karyogram]


def create_sequential_naming(converter: Optional[Callable[[int], NameType]]=None) -> Naming:
    gen = count()
    converter = converter if converter else lambda x: x

    def sequential_naming(generation: int, parents: Sequence[Individual]) -> NameType:
        """
        :param generation:
        :param parents:
        :return:
        """
        return converter(next(gen))
    return sequential_naming


def merge_karyograms(
        karyograms: Iterator[Karyogram],
        karyo_handler: Callable[[Iterator[Tuple[int, Iterator[Chromosome]]]], Karyogram]=dict,
        payload_handler: Callable[[Iterator[Chromosome]], Sequence[Chromosome]]=tuple
) -> Karyogram:
    """
    :param karyograms:
    :param karyo_handler:
    :param payload_handler:
    :return:
    >>> left = {
    ...     0: ({0: b'A', 1: b'A'},),
    ...     1: ({0: b'B', 1: b'B'},)
    ... }
    >>> right = {
    ...     0: ({0: b'C', 1: b'C'},),
    ...     1: ({0: b'D', 1: b'D'},)
    ... }
    >>> merge_karyograms((left, right)) # doctest: +ELLIPSIS
    {0: ({0: b'A', 1: b'A'}, {0: b'C', 1: b'C'}), 1: ({0: b'B', 1: b'B'}, {0: b'D', 1: b'D'})}
    """
    genetic_payload = sorted(
        (
            (pos, (chromosome for chromosome in chromosomes))
            for position in karyograms for (pos, chromosomes) in position.items()
        ),
        key=lambda x: x[0]
    )
    return karyo_handler(
        (pos, payload_handler(c for p in payload for c in p[1]))
        for (pos, payload) in groupby(genetic_payload, lambda x: x[0])
    )


def create_individual_builder(
        transcription: KaryoTranscription,
        naming: Naming,
        karyo_handler: Callable[[Iterator[Tuple[int, Iterator[Chromosome]]]], Karyogram]=dict,
        payload_handler: Callable[[Iterator[Chromosome]], Sequence[Chromosome]]=tuple,
        xover: Crossover=lambda x: x
) -> Spawning:
    """
    :param karyo_handler:
    :param payload_handler:
    :param transcription:
    :param naming:
    :return:
    >>> from pyvolution.types.gene import create_linear_mapping, create_chromosome_builder
    >>> mapping, remapping = create_linear_mapping(4)
    >>> builder = create_chromosome_builder(list, mapping, handle_gap=lambda x: b'')
    >>> spawner = create_individual_builder(builder, create_sequential_naming())
    >>> creature = spawner(('Hello', 'World'), 0)
    >>> creature.name
    0
    >>> creature.generation
    0
    >>> creature.karyogram # doctest: +ELLIPSIS
    {0: ({0: 'H', 1: 'e', 2: 'l', 3: 'l'}, {0: 'W', 1: 'o', 2: 'r', 3: 'l'}), 1: ({0: 'o'}, {0: 'd'})}
    """
    def spawn_individual(data: Iterator[DataType], generation: int) -> Individual:
        payload = sorted(
            ((pos, chromosome) for x in data for (pos, chromosome) in transcription(x).items()), key=lambda x: x[0]
        )
        karyogram = karyo_handler(
            (pos, payload_handler(chromosome[1] for chromosome in chromosomes))
            for (pos, chromosomes) in groupby(payload, key=lambda x: x[0])
        )

        return Individual(
            karyogram=xover(karyogram),
            generation=generation,
            name=naming(generation, tuple())
        )
    return spawn_individual


def select_half(karyogram: Karyogram) -> Karyogram:
    """
    :param karyogram:
    :return
    >>> k = {
    ...     0: ({0: b'H', 1: b'e', 2: b'l', 3: b'l'}, {0: b'W', 1: b'o', 2: b'r', 3: b'l'}),
    ...     1: ({0: b'o'}, {0: b'd'})
    ... }
    >>> selection = select_half(k)
    >>> selection[0][0] in k[0]
    True
    >>> selection[1][0] in k[1]
    True
    >>> select_half(dict())
    {}
    """
    return type(karyogram)(
        (pos, type(chromosomes)(choice(chromosomes) for _ in range(len(chromosomes) // 2)))
        for (pos, chromosomes) in karyogram.items()
    )


def create_gamete_builder(selector: Selector, xover: Crossover=lambda x: x) -> Mitosis:
    def create_gamete(individual: Individual) -> Gamete:
        return xover(selector(individual.karyogram))
    return create_gamete


def create_birth_builder(
        mitosis: Mitosis,
        naming: Naming,
        merge: Merging=merge_karyograms,
        xover: Crossover=lambda x: x
) -> Birthing:
    """
    :param mitosis:
    :param naming:
    :param merge:
    :return:
    >>> from pyvolution.types.gene import create_linear_mapping, create_chromosome_builder
    >>> mapping, remapping = create_linear_mapping(4)
    >>> builder = create_chromosome_builder(list, mapping, handle_gap=lambda x: b'')
    >>> spawner = create_individual_builder(builder, create_sequential_naming())
    >>> mitosis = create_gamete_builder(select_half)
    >>> birth = create_birth_builder(mitosis, create_sequential_naming())
    >>> parents = (spawner(('AAAACCCC', 'aaaacccc'), 0), spawner(('BBBBDDDD', 'bbbbdddd'), 0))
    >>> results = []
    >>> for i in range(1000):
    ...     child = birth(parents, 1)
    ...     one = ''.join(child.karyogram[0][0].values()), ''.join(child.karyogram[0][1].values())
    ...     two = ''.join(child.karyogram[1][0].values()), ''.join(child.karyogram[1][1].values())
    ...     results.append(one in [('AAAA', 'BBBB'), ('AAAA', 'bbbb'), ('aaaa', 'BBBB'), ('aaaa', 'bbbb')])
    ...     results.append(two in [('CCCC', 'DDDD'), ('CCCC', 'dddd'), ('cccc', 'DDDD'), ('cccc', 'dddd')])
    >>> all(results)
    True
    """
    def give_birth(parents: Sequence[Individual], generation: int) -> Individual:
        return Individual(
            karyogram=merge(xover(mitosis(parent)) for parent in parents),
            generation=generation,
            name=naming(generation, parents)
        )
    return give_birth
