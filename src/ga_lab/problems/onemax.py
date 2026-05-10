from __future__ import annotations

from ga_lab.problems.base import Fitness, Genome, ProblemMetadata


class OneMaxProblem:
    name = "onemax"
    compatible_representations = ("bit",)
    default_objective_directions = (True,)

    def __init__(
        self,
        family: str = "onemax",
        trap_block_size: int = 4,
        jump_k: int = 4,
    ) -> None:
        normalized = family.strip().lower().replace("-", "_")
        if normalized == "deceptive_trap":
            normalized = "trap"
        if normalized == "jump_k":
            normalized = "jump"
        if normalized not in {"onemax", "leading_ones", "trap", "jump"}:
            raise ValueError(
                "family must be one of: onemax, leading_ones, trap, jump"
            )
        if trap_block_size < 2:
            raise ValueError("trap_block_size must be >= 2")
        if jump_k < 2:
            raise ValueError("jump_k must be >= 2")
        self.family = normalized
        self.trap_block_size = int(trap_block_size)
        self.jump_k = int(jump_k)

    def _decode_bits(self, genome: Genome) -> list[int]:
        return [1 if float(gene) >= 0.5 else 0 for gene in genome]

    def _leading_ones(self, bits: list[int]) -> float:
        count = 0
        for bit in bits:
            if bit != 1:
                break
            count += 1
        return float(count)

    def _deceptive_trap(self, bits: list[int]) -> float:
        total = 0.0
        block_size = self.trap_block_size
        for start in range(0, len(bits), block_size):
            block = bits[start : start + block_size]
            if len(block) < block_size:
                total += float(sum(block))
                continue
            ones = sum(block)
            total += float(block_size if ones == block_size else (block_size - 1 - ones))
        return total

    def _jump(self, bits: list[int]) -> float:
        ones = sum(bits)
        length = len(bits)
        if ones == length:
            return float(length)
        if ones <= length - self.jump_k:
            return float(ones)
        return float(length - ones)

    def fitness(self, genome: Genome) -> Fitness:
        bits = self._decode_bits(genome)
        if self.family == "leading_ones":
            return self._leading_ones(bits)
        if self.family == "trap":
            return self._deceptive_trap(bits)
        if self.family == "jump":
            return self._jump(bits)
        return float(sum(bits))

    def optimal_fitness(self, genome_length: int) -> float:
        return float(genome_length)

    def metadata(self) -> ProblemMetadata:
        return ProblemMetadata(
            name=self.name,
            compatible_representations=self.compatible_representations,
            min_genome_length=1,
            default_objective_directions=self.default_objective_directions,
        )
