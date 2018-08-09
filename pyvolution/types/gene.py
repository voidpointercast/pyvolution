from typing import TypeVar, Sequence, Callable, Optional, Mapping, MutableMapping, Iterable, Tuple
from itertools import groupby
from collections import defaultdict
from attr import attrs, attrib


GeneType = TypeVar('GeneType')
DataType = TypeVar('DataType')
BaseType = TypeVar('BaseType')


@attrs
class GenePosition:
    chromosome: int = attrib()
    position: int = attrib()
    genome: GeneType = attrib()
    name: str = attrib(default='')

Chromosome = MutableMapping[int, BaseType]
ChromosomeSet = Mapping[int, Chromosome]
GeneMapping = Callable[[int], Tuple[int, int]]
GeneRemapping = Callable[[Tuple[int, int]], Optional[int]]
GeneEncoding = Callable[[GeneType], BaseType]
GeneDecoding = Callable[[BaseType], GeneType]
Transcription = Callable[[DataType], Sequence[GeneType]]
ReverseTranscription = Callable[[Sequence[GeneType]], DataType]
Karyogram = Mapping[int, Sequence[Chromosome]]
Dominance = Callable[[Iterable[GeneType]], GeneType]
KaryoTranscription = Callable[[DataType], Mapping[int, Chromosome]]
Crossover = Callable[[Karyogram], Karyogram]


def merge_chromosome_sets(sets: Sequence[ChromosomeSet]) -> Karyogram:
    # noinspection PyTypeChecker
    """
        :param sets:
        :return:
        >>> from pprint import pprint
        >>> first = {0: {0: b'H', 1: b'e', 2: b'l', 3: b'l'}, 1: {0: b'o', 1: b' ', 2: b'W', 3: b'o'}}
        >>> second = {0: {0: b'h', 1: b'E', 2: b'L', 3: b'L'}, 1: {0: b'O', 1: b' ', 2: b'w', 3: b'O'}}
        >>> pprint(dict(merge_chromosome_sets((first, second))))
        {0: [{0: b'H', 1: b'e', 2: b'l', 3: b'l'},
             {0: b'h', 1: b'E', 2: b'L', 3: b'L'}],
         1: [{0: b'o', 1: b' ', 2: b'W', 3: b'o'},
             {0: b'O', 1: b' ', 2: b'w', 3: b'O'}]}
        """
    karyogram = defaultdict(list)
    for cset in sets:
        for (index, chromosome) in cset.items():
            karyogram[index].append(chromosome)
    return karyogram


def remap_genome(
        remapping: GeneRemapping,
        decoding: GeneDecoding,
        retranscribe: ReverseTranscription,
        dominance: Dominance,
        karyogram: Karyogram
) -> DataType:
    # noinspection PyTypeChecker
    """
        :param remapping:
        :param decoding:
        :param retranscribe:
        :param dominance:
        :param karyogram:
        :return:
        >>> from pyvolution.types.gene import create_linear_mapping
        >>> mapping, remapping = create_linear_mapping(4)
        >>> builder = create_chromosome_builder(list, mapping, str.encode, handle_gap=lambda x: b'G')
        >>> first, second = builder('Hello World!'), builder('HELLO WORLD!')
        >>> remap_genome(remapping, bytes.decode, ''.join, max, merge_chromosome_sets((first, second)))
        'Hello World!'
        """
    decoded = sorted(
        ((cindex, gindex), decoding(base))
        for (cindex, chromosomes) in karyogram.items()
        for chromosome in chromosomes
        for (gindex, base) in chromosome.items()
    )
    dominant = tuple(zip(*sorted(
        (remapping(position), dominance(gene[1] for gene in genes))
        for position, genes in groupby(decoded, key=lambda x: x[0])
    )))
    return retranscribe(dominant[1])


def create_linear_mapping(
    chromosome_size: int
) -> Tuple[GeneMapping, GeneRemapping]:
    """
    :param chromosome_size:
    :return:
    >>> from random import randint
    >>> mapping, remapping = create_linear_mapping(randint(2, 10))
    >>> all(remapping(mapping(i)) == i for i in range(1000))
    True
    """
    def linear_mapping(position: int) -> Tuple[int, int]:
        return position // chromosome_size, position % chromosome_size

    def linear_remapping(position: Tuple[int, int]) -> int:
        return position[0] * chromosome_size + position[1]

    return linear_mapping, linear_remapping


def create_chromosome_builder(
        transcription: Transcription,
        mapping: GeneMapping,
        encoding: GeneEncoding,
        create_chromosome: Callable[[None], Chromosome]=dict,
        handle_gap: [Callable[[int], BaseType]]=lambda pos: None
) -> KaryoTranscription:
    # noinspection PyTypeChecker
    """
        :param transcription:
        :param mapping:
        :param encoding:
        :param create_chromosome:
        :param handle_gap:
        :return:
        >>> mapping, remapping = create_linear_mapping(4)
        >>> builder = create_chromosome_builder(list, mapping, str.encode, handle_gap=lambda x: b'G')
        >>> repr(dict(builder("Hello World!"))).replace(' ', '')
        "{0:{0:b'H',1:b'e',2:b'l',3:b'l'},1:{0:b'o',1:b'',2:b'W',3:b'o'},2:{0:b'r',1:b'l',2:b'd',3:b'!'}}"
        """
    def build_chromosome_set(data: DataType) -> ChromosomeSet:
        karyogram = defaultdict(create_chromosome)
        genes = transcription(data)
        for (i, gene) in enumerate(genes):
            chromosome, position = mapping(i)
            karyogram[chromosome][position] = encoding(gene)
        for chromosome in karyogram.values():
            gaps = set(range(max(karyogram.keys()))).difference(karyogram.keys())
            for gap in gaps:
                chromosome[gap] = handle_gap(gap)
        return karyogram
    return build_chromosome_set
