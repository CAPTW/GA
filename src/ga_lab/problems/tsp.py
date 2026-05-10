from __future__ import annotations

import math
import random

from ga_lab.problems.base import Fitness, Genome, ProblemMetadata


class TSPProblem:
    name = "tsp"
    compatible_representations = ("permutation",)
    default_objective_directions = (True,)

    def __init__(
        self,
        num_cities: int = 20,
        seed: int = 0,
        coordinates: list[list[float]] | None = None,
        distance_matrix: list[list[float]] | None = None,
        instance_name: str | None = None,
        instance_source: str | None = None,
        bounds: tuple[float, float] = (0.0, 100.0),
        return_to_start: bool = True,
    ) -> None:
        if num_cities <= 1:
            raise ValueError("num_cities must be > 1")
        self.num_cities = int(num_cities)
        self.return_to_start = return_to_start
        self.instance_name = instance_name
        self.instance_source = instance_source
        self._distance_matrix: list[list[float]] | None = None
        self.coordinates: list[tuple[float, float]] | None = None
        if coordinates is not None and distance_matrix is not None:
            raise ValueError("coordinates and distance_matrix cannot both be provided")
        if distance_matrix is not None:
            if len(distance_matrix) != self.num_cities:
                raise ValueError("distance_matrix row count must match num_cities")
            normalized_matrix: list[list[float]] = []
            for row in distance_matrix:
                if len(row) != self.num_cities:
                    raise ValueError("distance_matrix must be square with num_cities columns")
                normalized_matrix.append([float(value) for value in row])
            self._distance_matrix = normalized_matrix
        elif coordinates is not None:
            self.coordinates = [tuple(point) for point in coordinates]
            if len(self.coordinates) != self.num_cities:
                raise ValueError("coordinates length must match num_cities")
        else:
            low, high = bounds
            rng = random.Random(seed)
            self.coordinates = [
                (
                    rng.uniform(min(low, high), max(low, high)),
                    rng.uniform(min(low, high), max(low, high)),
                )
                for _ in range(self.num_cities)
            ]

    def _distance(self, city_a_idx: int, city_b_idx: int) -> float:
        if self._distance_matrix is not None:
            return self._distance_matrix[city_a_idx][city_b_idx]
        x1, y1 = self.coordinates[city_a_idx]
        x2, y2 = self.coordinates[city_b_idx]
        return math.hypot(x2 - x1, y2 - y1)

    def decode_route(self, genome: Genome) -> list[int]:
        route = [int(round(gene)) for gene in genome]
        if sorted(route) != list(range(self.num_cities)):
            raise ValueError("Genome must be a permutation of cities for TSP")
        return route

    def route_distance(self, route: list[int]) -> float:
        total = 0.0
        for idx in range(self.num_cities - 1):
            total += self._distance(route[idx], route[idx + 1])
        if self.return_to_start:
            total += self._distance(route[-1], route[0])
        return total

    def fitness(self, genome: Genome) -> Fitness:
        route = self.decode_route(genome)
        total = self.route_distance(route)
        # Return negative distance so larger is better with default maximize=true.
        return -total

    def solution_metrics(self, genome: Genome) -> dict[str, object]:
        route = self.decode_route(genome)
        cycle = route + [route[0]] if self.return_to_start else route[:]
        return {
            "best_route": route,
            "best_route_cycle": cycle,
            "best_route_distance": self.route_distance(route),
        }

    def optimal_fitness(self, genome_length: int) -> float | None:
        if genome_length != self.num_cities:
            return None
        return 0.0

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            exact_genome_length=self.num_cities,
            default_objective_directions=self.default_objective_directions,
        )
