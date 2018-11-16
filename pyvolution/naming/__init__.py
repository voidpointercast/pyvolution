from typing import Sequence
from itertools import count
from pyvolution.types.individual import Naming, NameType, Individual


def create_default_naming() -> Naming:
    id_gen = count()
    def default_naming(generation: int, parents: Sequence[Individual]) -> NameType:
        return dict(
            parents=[parent.name['id'] for parent in parents],
            id=next(id_gen)
        )
    return default_naming
