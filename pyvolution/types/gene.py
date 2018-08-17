from typing import TypeVar, Sequence, Callable, Optional, Mapping, MutableMapping, Iterable, Tuple
from functools import reduce
from operator import add
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

Chromosome = MutableMapping[int, GeneType]
ChromosomeSet = Mapping[int, Chromosome]
GeneMapping = Callable[[int], Tuple[int, int]]
GeneRemapping = Callable[[Tuple[int, int]], Optional[int]]
GeneEncoding = Callable[[Sequence[GeneType]], BaseType]
GeneDecoding = Callable[[BaseType], Sequence[GeneType]]
Transcription = Callable[[DataType], Sequence[GeneType]]
ReverseTranscription = Callable[[Sequence[GeneType]], DataType]
Karyogram = Mapping[int, Sequence[Chromosome]]
Dominance = Callable[[Iterable[GeneType]], GeneType]
KaryoTranscription = Callable[[DataType], Mapping[int, Chromosome]]
Crossover = Callable[[Karyogram], Karyogram]
Anomaly = Callable[[Karyogram], Karyogram]


def default_reduction(bases: Sequence[BaseType]) -> BaseType:
    return reduce(add, bases[1:], bases[0])


def create_crossover(
        encoding: GeneEncoding,
        decoding: GeneDecoding,
        selection: Callable[[Sequence[Chromosome]], Sequence[Tuple[Optional[Chromosome], Chromosome]]],
        indicator: Callable[[None], bool],
        reduction: Callable[[Sequence[BaseType]], BaseType]=default_reduction,
) -> Crossover:
    """
    :param reduction:
    :param encoding:
    :param decoding:
    :param selection:
    :param indicator:
    :return:
    >>> from random import uniform
    >>> karyogram = {0: [dict((i, i) for i in range(5)), dict((i, i) for i in range(5, 9))]}
    >>> def encoding(x: Sequence[int]) -> str:
    ...     return ''.join(str(num) + '#' for num in x)
    >>> def decoding(x: str) -> Sequence[int]:
    ...     return [int(num) for num in x.split('#') if num]
    >>> indicator = lambda: uniform(-1.0, 1.0) > 0.0
    >>> xover = create_crossover(encoding, decoding, lambda x: [(x[0], x[1])], indicator)
    >>> {num for c in xover(karyogram)[0] for num in c.values()}
    {0, 1, 2, 3, 5, 6, 7, 8}
    """
    def do_xover(left: Optional[Chromosome], right: Chromosome) -> Sequence[Chromosome]:
        if left is None:
            return [right]

        crossed_left, crossed_right = zip(
            *(
                (l, r) if indicator() else (r, l)
                for (l, r) in zip(encoding(left.values()), encoding(right.values()))
            )
        )
        return (
            type(left)(enumerate(decoding(reduction(crossed_left)))),
            type(right)(enumerate(decoding(reduction(crossed_right))))
        )

    def xover(karyogram: Karyogram) -> Karyogram:
        pairings = (
            (position, selection(chromosomes), type(chromosomes))
            for (position, chromosomes) in karyogram.items()
        )
        return type(karyogram)(
            (position, ctype(c for (left, right) in tuples for c in do_xover(left, right)))
            for (position, tuples, ctype) in pairings
        )
    return xover





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
        retranscribe: ReverseTranscription,
        dominance: Dominance,
        karyogram: Karyogram
) -> DataType:
    # noinspection PyTypeChecker
    """
        :param remapping:
        :param retranscribe:
        :param dominance:
        :param karyogram:
        :return:
        >>> from pyvolution.types.gene import create_linear_mapping
        >>> mapping, remapping = create_linear_mapping(4)
        >>> builder = create_chromosome_builder(list, mapping, handle_gap=lambda x: b'G')
        >>> first, second = builder('Hello World!'), builder('HELLO WORLD!')
        >>> remap_genome(remapping, ''.join, max, merge_chromosome_sets((first, second)))
        'Hello World!'
        """
    decoded = sorted(
        ((cindex, gindex), gene)
        for (cindex, chromosomes) in karyogram.items()
        for chromosome in chromosomes
        for (gindex, gene) in chromosome.items()
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
        >>> builder = create_chromosome_builder(list, mapping, handle_gap=lambda x: b'G')
        >>> repr(dict(builder("Hello World!"))).replace(' ', '')
        "{0:{0:'H',1:'e',2:'l',3:'l'},1:{0:'o',1:'',2:'W',3:'o'},2:{0:'r',1:'l',2:'d',3:'!'}}"
        """
    def build_chromosome_set(data: DataType) -> ChromosomeSet:
        karyogram = defaultdict(create_chromosome)
        genes = transcription(data)
        for (i, gene) in enumerate(genes):
            chromosome, position = mapping(i)
            karyogram[chromosome][position] = gene
        for chromosome in karyogram.values():
            gaps = set(range(max(karyogram.keys()))).difference(karyogram.keys())
            for gap in gaps:
                chromosome[gap] = handle_gap(gap)
        return karyogram
    return build_chromosome_set
