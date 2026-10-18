"""
network.py

The neural network that the genetic algorithm evolves.

It is NOT trained with backpropagation/gradient descent. Instead it is a
pure function of its weights: given a board position it outputs a score
in [-1, 1] saying how good the position is for White (positive = good
for White, negative = good for Black). The genetic algorithm searches
directly in weight-space for networks that produce good evaluations,
by having them play chess against each other.
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

# 12 piece planes (6 piece types x 2 colors) x 64 squares = 768
# + side to move (1) + castling rights x4 (white kingside/queenside,
#   black kingside/queenside) = 773
INPUT_SIZE = 773


class ChessEvalNet(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_sizes=(128, 64, 32)):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Tanh())  # squash to [-1, 1]
        self.net = nn.Sequential(*layers)

        # GA-evolved weights don't need gradients — saves memory/time.
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def evaluate(self, feature_vector: torch.Tensor) -> float:
        """feature_vector: shape (INPUT_SIZE,) -> python float score."""
        if feature_vector.dim() == 1:
            feature_vector = feature_vector.unsqueeze(0)
        return self.forward(feature_vector).item()
