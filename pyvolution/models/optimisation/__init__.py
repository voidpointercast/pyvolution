from typing import Callable, Sequence
from attr import attrib, attrs


@attrs
class TestFunction:
    function: Callable[[Sequence[float]], float] = attrib()
    arity: int = attrib(default=2)

    def __call__(self, *args: float) -> float:
        return self.function(args)

