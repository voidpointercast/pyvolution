from typing import Sequence, Tuple
from unittest import TestCase
from json import dump
from pyvolution.models.algebra.basic import create_basic_model, create_default_mutator, show_expression, BasicAlgebraJSONEncoder


class AlgebraicModelTest(TestCase):


    def test_monotony(self):
        popsize = 100
        mutator = create_default_mutator(0.2, 0.3)
        population, evolution, to_expression = create_basic_model(
            tuple((x, y, x**2 + y**2 + 1) for (x, y) in zip(range(-10, 10), range(-10, 10))),
            popsize=popsize
        )

        self.score = -float('infinity')
        def generate_score_invariant():
            def score_invariant(_, scores) -> bool:
                maximum = max(scores)
                valid = maximum >= self.score
                self.score = max(maximum, self.score)
                return valid
            return score_invariant


        scorings, bests, populations = self.perform_evolution(
            population,
            evolution,
            1000,
            mutator=mutator,
            popsize_invariant=lambda p, s: len(p) == popsize,
            score_invariant=generate_score_invariant()
        )
        print(show_expression(to_expression(bests[-1])), max(scorings[-1]))


    @classmethod
    def create_model(cls, points: Sequence[Tuple[float, float]], **kwargs):
        return create_basic_model(points, **kwargs)

    def perform_evolution(self, population, evolution, iterations: int=1000, mutator=create_default_mutator(), **kwargs):
        best_individuals = []
        scorings = []
        populations = (None, population)
        for i in range(iterations):
            population = tuple(evolution(population, i, 1, mutator))
            populations = populations[1], population
            best_individual, _ = max(population, key=lambda x: x[1])
            best_individuals.append(best_individual)
            population, scores = zip(*population)
            for (key, func) in kwargs.items():
                if not func(population, scores):
                    self.create_dump(key, populations, scores)
                    self.assertTrue(func(population, scores), msg='{0} violated.'.format(key))


            scorings.append(scores)
        return scorings, best_individuals, populations

    def create_dump(self, name: str, populations, scores):
        with open('{0}.dump.json'.format(name), 'w') as out:
            out.write(
                BasicAlgebraJSONEncoder().encode(
                    dict(
                        former_population=populations[0],
                        current_population=populations[1],
                        scores=scores
                    ),
                )
            )