from typing import Iterable, Sequence, Callable
from pyvolution.types.individual import NameType, Individual
from pyvolution.analysis import GenerationLoader

GenerationIterator = Iterable[Sequence[Individual]]
LineageFinder = Callable[[GenerationIterator], Iterable[Individual]]


def build_lineage_finder(predicate: Callable[[Individual], bool]) -> LineageFinder:
    def find_lineage(generations: GenerationIterator) -> Iterable[Individual]:
        return (
            individual
            for generation in generations
            for individual in generation
            if predicate(individual)
        )

    return find_lineage


def create_children_lineage(parent_id: int) -> Callable[[Individual], bool]:
    parents = [parent_id]
    def is_child_in_line(individual: Individual) -> bool:
        if individual.name['id'] in parents:
            return True

        for anchestor in parents:
            if anchestor in individual.name['parents']:
                parents.append(individual.name['id'])
                return True
        return False

    return is_child_in_line



def load_linage(predicate: Callable[[Individual], bool], data: GenerationIterator) -> Iterable[Individual]:
    """
    :param predicate:
    :param data:
    :return:
    >>> from pyvolution.analysis import create_yaml_loader
    >>> loader = create_yaml_loader('.', False)
    >>> lineage = load_linage(create_children_lineage(64), loader)
    >>> line = [child for child in lineage]
    >>> line
    """
    finder = build_lineage_finder(predicate)
    return finder(data)


