from __future__ import annotations

import random
from statistics import mean

from ga_lab.problems.base import Fitness, Genome, ProblemMetadata


class KnapsackProblem:
    name = "knapsack"
    compatible_representations = ("bit",)
    default_objective_directions = (True,)

    def __init__(
        self,
        num_items: int = 20,
        seed: int = 0,
        weights: list[float] | None = None,
        values: list[float] | None = None,
        capacity: float | None = None,
        instance_name: str | None = None,
        instance_source: str | None = None,
        weight_scale: tuple[float, float] = (1.0, 20.0),
        value_scale: tuple[float, float] = (1.0, 30.0),
        penalty_factor: float = 1000.0,
    ) -> None:
        if num_items <= 0:
            raise ValueError("num_items must be > 0")
        self.num_items = int(num_items)
        rng = random.Random(seed)
        if weights is None:
            min_w, max_w = weight_scale
            self.weights = [rng.uniform(min_w, max_w) for _ in range(self.num_items)]
        else:
            if len(weights) != self.num_items:
                raise ValueError("weights length must match num_items")
            self.weights = [float(w) for w in weights]
        if values is None:
            min_v, max_v = value_scale
            self.values = [rng.uniform(min_v, max_v) for _ in range(self.num_items)]
        else:
            if len(values) != self.num_items:
                raise ValueError("values length must match num_items")
            self.values = [float(v) for v in values]

        total_weight = sum(self.weights)
        if capacity is None:
            self.capacity = 0.5 * total_weight
        else:
            self.capacity = float(capacity)
        self.penalty_factor = float(penalty_factor)
        self.instance_name = instance_name
        self.instance_source = instance_source

    def decode_items(self, genome: Genome) -> list[int]:
        bits = [0 if gene <= 0 else 1 if gene >= 1 else int(round(gene)) for gene in genome]
        if len(bits) != self.num_items:
            raise ValueError("Genome length must match number of items for knapsack")
        return bits

    def evaluate_selection(self, bits: list[int]) -> dict[str, float]:
        total_weight = sum(
            weight for selected, weight in zip(bits, self.weights, strict=True) if selected == 1
        )
        total_value = sum(
            value for selected, value in zip(bits, self.values, strict=True) if selected == 1
        )
        constraint_violation = max(0.0, total_weight - self.capacity)
        constraint_violation_rate = (
            constraint_violation / self.capacity if self.capacity > 0 else 0.0
        )
        return {
            "total_weight": float(total_weight),
            "total_value": float(total_value),
            "constraint_violation": float(constraint_violation),
            "constraint_violation_rate": float(constraint_violation_rate),
        }

    def fitness(self, genome: Genome) -> Fitness:
        bits = self.decode_items(genome)
        selection = self.evaluate_selection(bits)
        penalty = self.penalty_factor * selection["constraint_violation"]
        return float(selection["total_value"] - penalty)

    def solution_metrics(self, genome: Genome) -> dict[str, object]:
        bits = self.decode_items(genome)
        selection = self.evaluate_selection(bits)
        selected_items = [idx for idx, selected in enumerate(bits) if selected == 1]
        return {
            "capacity": self.capacity,
            "best_selected_mask": bits,
            "best_selected_items": selected_items,
            "best_selected_count": len(selected_items),
            "best_total_weight": selection["total_weight"],
            "best_total_value": selection["total_value"],
            "best_constraint_violation": selection["constraint_violation"],
            "best_constraint_violation_rate": selection["constraint_violation_rate"],
            "best_is_feasible": selection["constraint_violation"] == 0.0,
        }

    def population_metrics(self, population: list[Genome]) -> dict[str, float]:
        if not population:
            return {}
        violation_rates: list[float] = []
        violations: list[float] = []
        infeasible_count = 0
        for genome in population:
            selection = self.evaluate_selection(self.decode_items(genome))
            violations.append(selection["constraint_violation"])
            violation_rates.append(selection["constraint_violation_rate"])
            if selection["constraint_violation"] > 0.0:
                infeasible_count += 1
        return {
            "population_constraint_violation_ratio": infeasible_count / len(population),
            "population_mean_constraint_violation": mean(violations),
            "population_mean_constraint_violation_rate": mean(violation_rates),
        }

    def optimal_fitness(self, genome_length: int) -> float | None:
        if genome_length != self.num_items:
            return None
        return sum(self.values)

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            exact_genome_length=self.num_items,
            default_objective_directions=self.default_objective_directions,
        )
