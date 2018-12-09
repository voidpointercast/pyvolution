from typing import Optional, Generic, TypeVar, Callable, Iterable, MutableMapping, Dict, TextIO, Sequence, Any, Mapping
from yaml import load_all
from operator import and_, or_
from attr import attrs, attrib, Factory
from pyvolution.types.gene import ChromosomeSet


T = TypeVar('T')
DEFAULT_TYPEMAP = dict(int=int, float=float, string=str, bool=bool, complex=complex)
DEFAULT_DOMINANCE = {'sum': sum, 'and': and_, 'or': or_, 'min': min, 'max': max}
XOVERS = {'default': lambda x: x}


@attrs
class GeneLayout(Generic[T]):
    type: type = attrib()
    label: Optional[str] = attrib(default=None)
    dominance: Callable[[Iterable[T]], T] = attrib(default=sum)
    mutate: bool = attrib(default=True)

@attrs
class ChromosomeLayout:
    name: Optional[str] = attrib(default=None)
    genes: Mapping[int, GeneLayout]=attrib(default=Factory(dict))
    xover: Callable[[ChromosomeSet], ChromosomeSet] = attrib(default=lambda x: x)


@attrs
class FlexibleLayout:
    chromosomes: MutableMapping[int, ChromosomeLayout] = attrib(default=Factory(dict))

def load_genes(
        typemap: Mapping[str, type],
        dominances: Mapping[str, Callable[[Iterable[T]], T]],
        genes: Sequence[Dict[str, Any]]
) -> Dict[int, GeneLayout]:
    gene_map = dict()
    current_position = 0
    for gene in genes:
        position = gene.get('position', current_position)
        gene_map[position] = GeneLayout(
            typemap[gene['type']],
            gene.get('label'),
            dominances[gene.get('dominance', 'default')],
            gene.get('mutable', True)
        )
        current_position = position + 1
    return gene_map



def load_flexible_layout(
        buffer: TextIO,
        types: Optional[Dict[str, type]]=None,
        dominance: Optional[Dict[str, Callable[[Iterable[T]], T]]]=None,
        xover: Optional[Dict[str, Callable[[ChromosomeSet], ChromosomeSet]]]=None,
        loader: Callable[[TextIO], Sequence[Dict[str, Any]]]=load_all
) -> FlexibleLayout:
    """
    :param buffer:
    :param types:
    :param dominance:
    :param xover:
    :param loader:
    :return:
    >>> with open('example/constrain.yaml') as src:
    ...     layout = load_flexible_layout(src, xover={'by_gender': lambda x: x})
    >>> print(layout)
    """
    type_map = dict(DEFAULT_TYPEMAP, **types if types else dict())
    dominance_map = dict(DEFAULT_DOMINANCE, **dominance if dominance else dict())
    xover_map = dict(XOVERS, **xover if xover else dict())
    layout: FlexibleLayout = FlexibleLayout()
    for (cid, chromosome) in enumerate(loader(buffer)):
        layout.chromosomes[cid] = ChromosomeLayout(
            chromosome.get('name'),
            load_genes(type_map, dominance_map, chromosome['genes']),
            xover_map[chromosome.get('xover', 'default')]
        )
    return layout


