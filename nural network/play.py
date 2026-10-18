"""
play.py

Play against the best evolved network from the console, using UCI
move notation (e.g. "e2e4", "g1f3", "e7e8q" for promotion).

Usage:
    python play.py --checkpoint checkpoints/best_final.pt --side white
"""

import argparse

# pyrefly: ignore [missing-import]
import chess
# pyrefly: ignore [missing-import]
import torch

from network import ChessEvalNet
from chess_env import choose_move, NetEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--side", choices=["white", "black"], default="white",
                         help="which side the human plays")
    args = parser.parse_args()

    net = ChessEvalNet()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    evaluator = NetEvaluator(net)

    board = chess.Board()
    human_is_white = args.side == "white"

    print("Enter moves in UCI format (e.g. e2e4). Type 'quit' to exit.\n")

    while not board.is_game_over():
        print(board)
        print()
        human_turn = (board.turn == chess.WHITE) == human_is_white

        if human_turn:
            move = None
            while move is None:
                user_input = input("Your move: ").strip()
                if user_input.lower() == "quit":
                    return
                try:
                    candidate = chess.Move.from_uci(user_input)
                    if candidate in board.legal_moves:
                        move = candidate
                    else:
                        print("Illegal move, try again.")
                except ValueError:
                    print("Couldn't parse that move, use UCI format like e2e4.")
            board.push(move)
        else:
            move = choose_move(evaluator, board)
            print(f"AI plays: {move.uci()}\n")
            board.push(move)

    print(board)
    print("\nGame over:", board.result())


if __name__ == "__main__":
    main()
