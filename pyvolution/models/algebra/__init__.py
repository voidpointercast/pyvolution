from typing import TypeVar, Sequence, Tuple, Callable, Generator, Dict, Optional, cast, Sized
from random import random, uniform, randint
from operator import add, sub, mul, truediv, pow
from enum import Enum
from attr import attrib, attrs
from pyvolution.types.gene import remap_genome


DomainType = TypeVar('DomainType')
CodomainType = TypeVar('CodomainType')
SeedType = TypeVar('SeedType')


@attrs
class Function:
    evaluation: Optional[Callable[[Sequence[DomainType]], CodomainType]] = attrib()
    args: int = attrib(default=2)
    name: str = attrib(default=None)


class DefaultSeedTypes(Enum):
    CONSTANT = 0
    VARIABLE = 1
    FUNCTION = 2

    @classmethod
    def map_modul(cls, value: int) -> 'DefaultSeedTypes':
        """
        :param value:
        :return:
        >>> DefaultSeedTypes.map_modul(4) == DefaultSeedTypes.VARIABLE
        True
        """
        return DefaultSeedTypes(value % len(cast(Sized, DefaultSeedTypes)))

    def __lt__(self, other) -> bool:
        return self.value < other.value


DefaultSeed = Tuple[DefaultSeedTypes, float]
Expression = Tuple[Function, Sequence['Expression']]
ExpressionParser = Callable[[Sequence[SeedType]], Expression]
Interpretation = Callable[[SeedType], Function]


DEFAULT_FUNCTIONS = tuple(
    Function(f, 2, f.__name__) for f in [add, sub, mul, truediv, pow]
)
DEFAULT_VARIABLES = tuple('xy')


def uniform_choice_from(bins: int, value: float) -> int:
    return int(value % bins)


def create_default_interpretation(
        functions: Sequence[Function]=DEFAULT_FUNCTIONS,
        variables: Sequence[str]=DEFAULT_VARIABLES
) -> Interpretation:
    def default_interpretation(seed: DefaultSeed) -> Function:
        func_type, value = seed
        if func_type == DefaultSeedTypes.FUNCTION:
            return functions[uniform_choice_from(len(functions), value)]
        if func_type == DefaultSeedTypes.CONSTANT:
            return Function(lambda: value, 0, 'CONSTANT')
        if func_type == DefaultSeedTypes.VARIABLE:
            return Function(None, 0, variables[uniform_choice_from(len(variables), value)])
    return default_interpretation


def create_expression_parser(
        interpretation: Interpretation=create_default_interpretation()) -> ExpressionParser:
    """
    :param interpretation:
    :return:
    >>> from pprint import pprint
    >>> seed = [(DefaultSeedTypes.FUNCTION, 0.0), (DefaultSeedTypes.VARIABLE, 1.0), (DefaultSeedTypes.VARIABLE, 3.0)]
    >>> parser = create_expression_parser()
    >>> pprint(parser(seed))
    (Function(evaluation=<built-in function add>, args=2, name='add'),
     ((Function(evaluation=None, args=0, name='y'), ()),
      (Function(evaluation=None, args=0, name='x'), ())))
    >>> parser(seed[0: 1])
    ()
    """
    def parse(data: Generator[SeedType, None, None]) -> Expression:
        f = interpretation(next(data))
        arguments = tuple(parse(data) for _ in range(f.args))
        if len(arguments) < f.args:
            raise StopIteration("No more data available")
        return f, arguments

    def parse_expression(seed: Sequence[SeedType]) -> Expression:
        try:
            return parse(token for token in seed)
        except StopIteration:
            return tuple()


    return parse_expression


def evaluate(
        expression: Expression,
        variables: Dict[str, DomainType],
        zero: DomainType,
        default: Optional[DomainType]=None
) -> CodomainType:
    """
    :param expression:
    :param variables:
    :param zero:
    :return:
    >>> seed = [(DefaultSeedTypes.FUNCTION, 0.0), (DefaultSeedTypes.CONSTANT, 1.0), (DefaultSeedTypes.CONSTANT, 2.0)]
    >>> parser = create_expression_parser()
    >>> evaluate(parser(seed), dict(), 0.0)
    3.0
    >>> seed = [
    ...     (DefaultSeedTypes.FUNCTION, 0.0),
    ...     (DefaultSeedTypes.CONSTANT, 1.0),
    ...     (DefaultSeedTypes.VARIABLE, 0.0)
    ... ]
    >>> evaluate(parser(seed), dict(x=10), 0)
    11.0
    >>> evaluate(parser([(DefaultSeedTypes.FUNCTION, 0.0)]), dict(), 0.0, -float('infinity'))
    -inf
    """

    if not expression:
        return default if default else zero
    func, args = expression
    if func.evaluation is None:
        return variables.get(func.name, zero)
    else:
        return func.evaluation(*(evaluate(arg, variables, zero) for arg in args))


def create_default_mutator(
        type_propability: float=0.1,
        value_propability: float=0.1
) -> Callable[[DefaultSeed], DefaultSeed]:
    def mutate_seed(seed: DefaultSeed) -> DefaultSeed:
        return (
            DefaultSeedTypes.map_modul(seed[0].value + (randint(-1, 1) if random() <= type_propability else 0)),
            seed[1] + (uniform(-1.0, 1.0) if random() <= value_propability else 0.0)
        )
    return mutate_seed


def show_expression(expression: Expression) -> str:
    f, args = expression
    if f.evaluation is None:
        return f.name
    if f.name == 'CONSTANT':
        return str(f.evaluation())
    return '{0}({1})'.format(f.name, ','.join(show_expression(arg) for arg in args))



def create_function_from_expression(
        expression: Expression,
        variable_names: Sequence[str],
        zero: DomainType,
        default: Optional[DomainType]=None
) -> Callable[[Sequence[DomainType]], CodomainType]:
    """
    :param expression:
    :param variable_names:
    :param zero:
    :param default:
    :return:
    >>> seed = [(DefaultSeedTypes.FUNCTION, 0.0), (DefaultSeedTypes.CONSTANT, 1.0), (DefaultSeedTypes.CONSTANT, 2.0)]
    >>> parser = create_expression_parser()
    >>> evaluate(parser(seed), dict(), 0.0)
    3.0
    >>> seed = [
    ...     (DefaultSeedTypes.FUNCTION, 0.0),
    ...     (DefaultSeedTypes.CONSTANT, 1.0),
    ...     (DefaultSeedTypes.VARIABLE, 0.0)
    ... ]
    >>> expr = parser(seed)
    >>> f = create_function_from_expression(expr, ['x'], 0)
    >>> f(2.0)
    3.0
    >>> f(x=4.0)
    5.0
    >>> f(2.0, x=3.0)
    4.0
    """
    def expression_func(*args, **kwargs) -> CodomainType:
        vars = dict(zip(variable_names, args), **kwargs)
        return evaluate(expression, vars, zero, default)

    return expression_func

