"""
train.py

Runs the genetic algorithm: evaluate fitness -> evolve -> repeat.
Saves the best network so far every few generations so you can watch
it improve, and so training can be resumed / inspected.

Usage:
    python train.py --generations 50 --population 24 --games 4
"""

import argparse
import os
import time

import torch

from genetic import Population


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--games", type=int, default=4,
                         help="games played per genome per generation")
    parser.add_argument("--mutation-rate", type=float, default=0.08)
    parser.add_argument("--mutation-strength", type=float, default=0.4)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pop = Population(
        size=args.population,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        elite_count=args.elite,
        seed=args.seed,
    )

    print(f"Starting evolution: population={args.population}, "
          f"generations={args.generations}, games/genome={args.games}")

    for gen in range(args.generations):
        start = time.time()
        pop.evaluate_fitness(games_per_genome=args.games, verbose=True)
        elapsed = time.time() - start
        print(f"    generation {gen} evaluated in {elapsed:.1f}s")

        if gen % args.save_every == 0 or gen == args.generations - 1:
            best = pop.best_net()
            path = os.path.join(args.out_dir, f"best_gen_{gen}.pt")
            torch.save(best.state_dict(), path)
            print(f"    saved {path}")

        pop.evolve_generation()

    final_best = pop.best_net()
    final_path = os.path.join(args.out_dir, "best_final.pt")
    torch.save(final_best.state_dict(), final_path)
    print(f"Training complete. Final best network saved to {final_path}")


if __name__ == "__main__":
    main()
