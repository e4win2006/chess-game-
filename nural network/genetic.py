"""
genetic.py

The evolutionary algorithm itself. A "genome" here is just the flat
vector of weights of a ChessEvalNet. Each generation:

  1. Fitness: every genome plays a handful of games (as White and as
     Black) against opponents drawn from the population, and against
     a simple material-counting baseline. Wins/draws/losses (plus a
     small material-based shaping bonus) become the fitness score.
  2. Selection: tournament selection favors fitter genomes, but not
     deterministically, to keep some diversity.
  3. Crossover: two parent weight-vectors are mixed to make a child.
  4. Mutation: gaussian noise is sprinkled into the child's weights.
  5. Elitism: the best few genomes are copied unchanged into the next
     generation so progress is never lost.
"""

import copy
import random

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
from network import ChessEvalNet
from chess_env import play_game, NetEvaluator, MaterialEvaluator


def net_to_vector(net: ChessEvalNet) -> np.ndarray:
    return np.concatenate([p.cpu().numpy().ravel() for p in net.parameters()])


def vector_to_net(vector: np.ndarray, template: ChessEvalNet) -> ChessEvalNet:
    net = copy.deepcopy(template)
    idx = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            chunk = vector[idx: idx + n].reshape(p.shape)
            p.copy_(torch.tensor(chunk, dtype=p.dtype))
            idx += n
    return net


class Population:
    def __init__(self, size=24, hidden_sizes=(128, 64, 32), mutation_rate=0.08,
                 mutation_strength=0.4, elite_count=2, seed=None):
        self.size = size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_count = elite_count
        self.template = ChessEvalNet(hidden_sizes=hidden_sizes)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.genomes = [net_to_vector(ChessEvalNet(hidden_sizes=hidden_sizes))
                         for _ in range(size)]
        self.fitness = [0.0] * size
        self.generation = 0

    def nets(self):
        return [vector_to_net(g, self.template) for g in self.genomes]

    def evaluate_fitness(self, games_per_genome=4, verbose=False):
        """Round-robin-ish: each genome plays `games_per_genome` games
        against randomly chosen opponents (half vs other genomes, half
        vs the material baseline), alternating colors."""
        evaluators = [NetEvaluator(n) for n in self.nets()]
        baseline = MaterialEvaluator()
        scores = [0.0] * self.size

        for i in range(self.size):
            opponents = random.sample(range(self.size), k=min(3, self.size - 1) or 1)
            opponents = [o for o in opponents if o != i][:games_per_genome // 2 or 1]

            games_played = 0
            for j in opponents:
                if games_played >= games_per_genome:
                    break
                as_white = play_game(evaluators[i], evaluators[j])
                scores[i] += self._score_from_result(as_white["result"], "1-0")
                scores[j] += self._score_from_result(as_white["result"], "0-1")

                as_black = play_game(evaluators[j], evaluators[i])
                scores[i] += self._score_from_result(as_black["result"], "0-1")
                scores[j] += self._score_from_result(as_black["result"], "1-0")
                games_played += 2

            # Remaining games vs the material baseline, to anchor
            # fitness to something resembling real chess skill.
            while games_played < games_per_genome:
                white_game = play_game(evaluators[i], baseline)
                scores[i] += self._score_from_result(white_game["result"], "1-0")
                scores[i] += 0.02 * white_game["material_diff"]
                games_played += 1

        self.fitness = scores
        if verbose:
            best = max(scores)
            avg = sum(scores) / len(scores)
            print(f"  Gen {self.generation}: best fitness={best:.2f}, avg={avg:.2f}")

    @staticmethod
    def _score_from_result(result: str, win_string: str) -> float:
        if result == win_string:
            return 1.5  # Reward for victory
        if result == "1/2-1/2":
            return 0.3  # Draw reward
        return -0.8     # Pain (penalty) for failure / defeat

    def _tournament_select(self, k=3) -> np.ndarray:
        contenders = random.sample(range(self.size), k)
        best = max(contenders, key=lambda idx: self.fitness[idx])
        return self.genomes[best]

    @staticmethod
    def _crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        mask = np.random.rand(len(parent_a)) < 0.5
        return np.where(mask, parent_a, parent_b)

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mutation_mask = np.random.rand(len(genome)) < self.mutation_rate
        noise = np.random.randn(len(genome)) * self.mutation_strength
        return genome + mutation_mask * noise

    def evolve_generation(self):
        ranked = sorted(range(self.size), key=lambda idx: self.fitness[idx], reverse=True)
        new_genomes = [self.genomes[idx].copy() for idx in ranked[:self.elite_count]]

        while len(new_genomes) < self.size:
            parent_a = self._tournament_select()
            parent_b = self._tournament_select()
            child = self._crossover(parent_a, parent_b)
            child = self._mutate(child)
            new_genomes.append(child)

        self.genomes = new_genomes
        self.generation += 1

    def best_net(self) -> ChessEvalNet:
        best_idx = max(range(self.size), key=lambda idx: self.fitness[idx])
        return vector_to_net(self.genomes[best_idx], self.template)
