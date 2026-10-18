# Evolving Chess Neural Network

A neural network that learns to play chess through **genetic evolution**
instead of backpropagation. A population of networks (random weights at
first) play chess against each other; the ones that win more get to
"breed" — their weights are mixed together and mutated to form the next
generation. Repeat for many generations and the population's chess
ability improves over time.

## How it works

- **`network.py`** — `ChessEvalNet`: a small feedforward network. Given
  a board position (encoded as a 773-length vector: 12 piece planes x
  64 squares + turn + castling rights) it outputs a score in `[-1, 1]`
  — how good the position looks for White.
- **`chess_env.py`** — uses [`python-chess`](https://python-chess.readthedocs.io/)
  for legal move generation and game rules. `choose_move()` does a
  1-ply lookahead: try every legal move, score the resulting position
  with the network, pick the best one. Also includes a fixed
  material-counting baseline opponent (`MaterialEvaluator`) so fitness
  is anchored to something resembling real chess skill, not just
  "beats other bad networks."
- **`genetic.py`** — the evolutionary algorithm:
  - **Fitness**: each genome plays several games (as White and Black)
    against other genomes and the material baseline. Win = 1 point,
    draw = 0.4, loss = 0, plus a small bonus for material advantage.
  - **Selection**: tournament selection (pick 3 at random, keep the
    fittest) — favors strong genomes without discarding diversity.
  - **Crossover**: a child's weights are randomly mixed 50/50 from two
    parents (uniform crossover).
  - **Mutation**: ~8% of a child's weights get random gaussian noise
    added, so the population keeps exploring new strategies.
  - **Elitism**: the top 2 genomes are copied unchanged into the next
    generation, so a good network is never accidentally lost.
- **`train.py`** — runs the full evolve loop for N generations,
  checkpointing the best network periodically.
- **`play.py`** — play against the best evolved network from the
  console using UCI move notation (e.g. `e2e4`).

## Setup

```bash
pip install torch python-chess numpy
```

## Train

```bash
python train.py --generations 50 --population 24 --games 4
```

Key flags:
- `--population` — how many networks per generation (bigger = more
  diversity, slower per generation)
- `--games` — games played per genome per generation (more = more
  reliable fitness estimate, slower)
- `--mutation-rate` / `--mutation-strength` — how much variation gets
  introduced each generation
- `--elite` — how many top genomes survive unchanged each generation
- `--save-every` — checkpoint frequency (saved to `checkpoints/`)

This is CPU-bound and each game is a full chess game with 1-ply
lookahead per move, so expect roughly 5-15 games/second depending on
your machine. A population of 24 playing 4 games each is ~48-96 games
per generation.

## Play against it

```bash
python play.py --checkpoint checkpoints/best_final.pt --side white
```

## Notes on scope / what this is (and isn't)

- This evolves an **evaluation function**, combined with a shallow
  (1-ply) search. It is not a full engine with deep alpha-beta search,
  and it isn't AlphaZero-style Monte Carlo Tree Search + self-play
  reinforcement learning — that's a different (heavier) approach if
  you want to go further.
- Early generations will play very weak, sometimes nonsensical chess.
  Expect it to take real time (many generations, decent population
  size) before you see coherent opening play or basic tactics emerge.
- Natural next steps if you want to push this further:
  - Increase lookahead depth (minimax/alpha-beta instead of 1-ply)
    for both fitness games and final play — much stronger play, but
    slower training.
  - Track an ELO-style rating across generations instead of raw
    win/loss, for a clearer signal of whether it's actually improving.
  - Evolve network *architecture* too (NEAT-style), not just weights.
  - Seed the initial population with some hand-crafted evaluation
    heuristics (piece-square tables) so it doesn't start from zero.
