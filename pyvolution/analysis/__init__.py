from typing import Callable, Sequence, TypeVar, Generator, Union, Tuple
from os import makedirs
from os.path import join, exists
from yaml import dump, load_all
from attr import asdict
from pyvolution.types.individual import Individual

SerialisedIndividual = TypeVar('SerialisedIndividual')
IndividualStorage = Callable[[int, Sequence[SerialisedIndividual]], None]
PopulationStorage = Callable[[int, Sequence[Union[Individual, Tuple[Individual, float]]]], None]
GenerationLoader = Callable[[None], Generator[Sequence[Individual], None, None]]


def create_yaml_store(basedir: str, name: str='result', split: bool=True) -> IndividualStorage:
    makedirs(basedir, exist_ok=True)
    def yaml_file_store(_: int, population: Sequence[SerialisedIndividual]) -> None:
        with open(join(basedir, '{0}.yaml'.format(name)), 'a') as out:
            dump(population, out, allow_unicode=True, explicit_start=True)

    def yaml_dir_store(generation: int, population: Sequence[SerialisedIndividual]) -> None:
        with open(join(basedir, '{1}_{0}.yaml'.format(generation, name)), 'w') as out:
            dump(population, allow_unicode=True)

    return yaml_dir_store if split else yaml_file_store


def create_yaml_loader(basedir: str, name: str='result', split: bool=True) -> Generator[Sequence[Individual], None, None]:
    if not split:
        with open(join(basedir, '{0}.yaml'.format(name))) as src:
            for generation in load_all(src):
                yield (Individual(**individual) for individual in generation)

    else:
        gen = 0
        while exists(join(basedir, '{1}_{0}.yaml'.format(gen, name))):
            with open(join(basedir, '{1}_{0}.yaml'.format(gen, name))) as src:
                yield (
                    Individual(**individual)
                    for individual in load_all(src)
                )


def create_serialisation_pattern(serialisation: Callable[[Individual], SerialisedIndividual]):
    def serialise(individual: Union[Individual, Tuple[Individual, float]]) -> SerialisedIndividual:
        if isinstance(individual, Individual):
            return serialisation(individual)
        else:
            result = serialisation(individual[0])
            result['meta']['fitness'] = individual[1]
            return result
    return serialise


def create_population_storage(
        store: IndividualStorage,
        serialisation: Callable[[Individual], SerialisedIndividual]=asdict
) -> PopulationStorage:
    """
    :param store:
    :param serialisation:
    :return:
    >>> from tempfile import TemporaryDirectory
    >>> from pyvolution.types.population import create_sample_population
    >>> from yaml import load_all
    >>> pop = list()
    >>> pop.extend(create_sample_population())
    >>> pop.extend(create_sample_population(generation=1))
    >>> with TemporaryDirectory() as temp:
    ...     store = create_population_storage(create_yaml_store(temp, False))
    ...     store(0, pop)
    ...     store(1, pop)
    ...     loader = create_yaml_loader(temp, False)
    ...     loaded = list(i for gen in loader for i in gen)
    ...     with open(join(temp, 'population.yaml')) as src:
    ...         zeros, ones= list(load_all(src))
    ...
    >>> len(zeros), len(ones)
    (10, 10)
    >>> loaded == pop
    True
    """


    def belongs_to_generation(
            generation: int,
            individual: Union[Individual, Tuple[Individual, float]]
    ) -> bool:
        if isinstance(individual, Individual):
            return individual.generation == generation
        else:
            return individual[0].generation == generation

    serialise = create_serialisation_pattern(serialisation)

    def store_generation(generation: int, population: Sequence[Individual]) -> None:
        generation_serialised = list(
            serialise(individual)
            for individual in population
            if belongs_to_generation(generation, individual)
        )
        store(generation, generation_serialised)

    return store_generation


def create_population_subset_store(
        predicate: Callable[[int, Individual], bool],
        store: IndividualStorage,
        serialisation: Callable[[Individual], SerialisedIndividual] = asdict
) -> PopulationStorage:
    serialise = create_serialisation_pattern(serialisation)
    def store_population_subset(generation: int, population: Sequence[Individual]) -> None:
        generation_serialised = list(
            serialise(individual)
            for individual in population
            if predicate(generation, individual)
        )
        store(generation, generation_serialised)
    return store_population_subset