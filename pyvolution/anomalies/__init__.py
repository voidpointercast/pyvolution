from typing import Callable
from pyvolution.types.gene import Anomaly, Chromosome, Karyogram, GeneEncoding, GeneDecoding, BaseType

ChromosomialAnomaly = Callable[[BaseType], BaseType]


def create_chromosomial_anomaly(
        encoding: GeneEncoding,
        decoding: GeneDecoding,
        anomaly: ChromosomialAnomaly
) -> Anomaly:
    """
    :param encoding:
    :param decoding:
    :param anomaly:
    :return:
    >>> from typing import Sequence
    >>> def anomaly(values: Sequence[int]) -> Sequence[int]: return list(values) + [0]
    >>> karyogram = {0: [dict(enumerate(range(10))) for _ in range(4)]}
    >>> application = create_chromosomial_anomaly(lambda x: x, lambda x: x, anomaly)
    >>> modified = application(karyogram)
    >>> all(len(x) == 11 for c in modified.values() for x in c)
    True
    """
    def apply_chromosomial_anomaly(karyogram: Karyogram) -> Karyogram:
        return type(karyogram)(
            (
                position,
                type(chromosomes)(
                    type(chromosome)(
                        enumerate(decoding(anomaly(encoding(chromosome.values())))))
                        for chromosome in chromosomes
                    )
            )
            for (position, chromosomes) in karyogram.items()
        )
    return apply_chromosomial_anomaly