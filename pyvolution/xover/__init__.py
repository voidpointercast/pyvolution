from typing import Callable, Iterator, Tuple, Any
from itertools import chain, cycle, zip_longest
from pyvolution.types.gene import Chromosome, Crossover


XoverSectionSelector = Callable[[Chromosome, Chromosome], Iterator[bool]]
Xover = Callable[[Chromosome, Chromosome], Tuple[Chromosome, Chromosome]]


def split_section_half(left: Chromosome, right: Chromosome) -> Iterator[bool]:
    """
    :param left:
    :param right:
    :return:
    >>> list(split_section_half(dict(zip(range(4), range(4))), dict(zip(range(4), range(4)))))
    [True, True, False, False]
    """
    mid = int(min(map(len, (left, right))) / 2)
    return (
        i < mid
        for i in range(max(map(len, (left, right))))
    )


def build_xover_application(selector: XoverSectionSelector) -> Xover:
    """
    :param selector:
    :return:
    >>> from pprint import pprint
    >>> left, right = dict(zip(range(10), range(10))), dict(zip(range(10), (x**2 for x in range(10))))
    >>> apply_xover = build_xover_application(split_section_half)
    >>> pprint(apply_xover(left, right))
    ({0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
     {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 25, 6: 36, 7: 49, 8: 64, 9: 81})
    >>> left, right = dict(zip(range(10), range(10))), dict(zip(range(12), (-x for x in range(12))))
    >>> new_left, new_right = apply_xover(left, right)
    >>> print(new_left)
    {0: 0, 1: -1, 2: -2, 3: -3, 4: -4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
    >>> print(new_right)
    {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: -5, 6: -6, 7: -7, 8: -8, 9: -9, 10: -10, 11: -11}
    """
    def not_empty(item: Tuple[int, Any]) -> bool: return item[0] is not None

    def apply_xover(left: Chromosome, right: Chromosome) -> Tuple[Chromosome, Chromosome]:
        sections = selector(left, right)
        xover_product = (
            ((ir, r) if xover else (il, l), (il, l) if xover else (ir, r))
            for (((il, l), (ir, r)), xover)
            in zip(zip_longest(left.items(), right.items(), fillvalue=(None, None)), chain(sections, cycle([False])))
        )
        new_left, new_right = zip(*xover_product)
        return type(left)(filter(not_empty, new_left)), type(right)(filter(not_empty, new_right))
    return apply_xover
