from typing import Set, Optional, Callable, Sequence
from math import log10, ceil
from matplotlib.pyplot import subplots, Figure, colorbar
from pyvolution.types.gene import GeneType, aggregate_chromosomes
from pyvolution.types.individual import Individual, Karyogram

def create_karyogram_plotter(
        gene_value,
        chromosome_indices: Optional[Set[int]]=None,
        colormap=None,
        **kwargs

) -> Callable[[Individual], Figure]:
    """
    :param gene_value:
    :param chromosome_indices:
    :return:
    >>> from matplotlib.pyplot import show
    >>> from pyvolution.types.individual import create_sample_individual
    >>> sample = create_sample_individual(3, 4, gene_space=range(10, 50))
    >>> plot = create_karyogram_plotter(lambda x: x)
    >>> figure = plot(sample)
    >>> figure.show()
    """


    def chromosome_filter(index: int) -> bool:
        return True if chromosome_indices is None else index in chromosome_indices


    def plot_karyogram(individual: Individual) -> Figure:
        figure, axis = subplots()
        karyogram: Karyogram = dict(
            (index, chromosomes)
            for (index, chromosomes) in individual.karyogram.items()
            if chromosome_filter(index)
        )

        data = list()
        for (index, chromosomes) in karyogram.items():
            cindex_multiplicator = 0.1 ** int(ceil(log10(len(chromosomes))))
            cindex_multiplicator = kwargs.get('override_gap_multiplicator', cindex_multiplicator)

            for (cindex, chromosome) in enumerate(chromosomes):
                y = index + cindex * cindex_multiplicator
                for (gindex, gene) in chromosome.items():
                    data.append((gindex, y, gene_value(gene)))

        xs, ys, vs = zip(*data)
        scatter = axis.scatter(xs, ys, vs, c=vs, cmap=colormap)
        colorbar(scatter)
        axis.set_title(individual.name)

        return figure

    return plot_karyogram


def create_karyogram_aggregate_plotter(
        gene_value,
        aggregate: Callable[[Sequence[GeneType]], GeneType],
        chromosome_indices: Optional[Set[int]]=None,
        colormap=None,
        **kwargs
) -> Callable[[Individual], Figure]:
    """
    :param gene_value:
    :param aggregate:
    :param chromosome_indices:
    :param colormap:
    :return:
    >>> from matplotlib.pyplot import show
    >>> from pyvolution.types.individual import create_sample_individual
    >>> sample = create_sample_individual(3, 4, gene_space=range(10, 50))
    >>> plot = create_karyogram_aggregate_plotter(lambda x: x, sum)
    >>> figure = plot(sample)
    >>> figure.show()
    """
    kwargs['override_gap_multiplicator'] = kwargs.get('override_gap_multiplicator', 0.0)
    plotter = create_karyogram_plotter(gene_value, chromosome_indices, colormap, **kwargs)


    def plot_aggregated_karyogram(individual: Individual) -> Figure:
        aggregated_karyogram = dict(
            (index, [aggregate_chromosomes(aggregate, chromosomes)])
            for (index, chromosomes) in individual.karyogram.items()
        )


        return plotter(
            Individual(aggregated_karyogram, individual.generation, individual.name))

    return plot_aggregated_karyogram
