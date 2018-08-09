from pyvolution.types.population import Fitness, SurvivalIndication, RankedPopulation


def create_threshold_indication(threshold: Fitness) -> SurvivalIndication:
    def threshold_indication(value: Fitness) -> bool:
        return value >= threshold
    return threshold_indication


def keep_best_halve(population: RankedPopulation) -> RankedPopulation:
    ranking = sorted(population, key=lambda x: x[1], reversed=True)
    return ranking[0: len(ranking) // 2]
