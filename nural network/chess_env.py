"""
chess_env.py

Bridges python-chess (board/rules/legal moves) with a ChessEvalNet
(position scoring). Provides:
  - encode_board(): board -> 773-dim feature tensor
  - choose_move(): 1-ply search using the net to score resulting positions
  - play_game(): plays two nets against each other, returns the result
"""

# pyrefly: ignore [missing-import]
import chess
# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
from network import ChessEvalNet, INPUT_SIZE

# Piece order for board encoding (matches network.py INPUT_SIZE layout)
PIECE_ORDER = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING,
]

# Material values used by MaterialEvaluator and material_balance()
# (defined here, before any function that references them)
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}


# ─── Board encoder ────────────────────────────────────────────────────────────

def encode_board(board: chess.Board) -> torch.Tensor:
    """Turn a python-chess Board into a 773-dim float tensor."""
    vec = torch.zeros(INPUT_SIZE, dtype=torch.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        piece_idx = PIECE_ORDER.index(piece.piece_type)
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + piece_idx
        vec[plane * 64 + square] = 1.0

    base = 768
    vec[base + 0] = 1.0 if board.turn == chess.WHITE else 0.0
    vec[base + 1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    vec[base + 2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    vec[base + 3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    vec[base + 4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    return vec


# ─── Material balance (used by MaterialEvaluator and play_game) ───────────────

def material_balance(board: chess.Board) -> float:
    """Positive = White ahead on material."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total -= value * len(board.pieces(piece_type, chess.BLACK))
    return total


# ─── Evaluator classes ────────────────────────────────────────────────────────

class NetEvaluator:
    """Wraps a ChessEvalNet so it can score a board position directly."""

    def __init__(self, net: ChessEvalNet):
        self.net = net

    def score(self, board: chess.Board) -> float:
        return self.net.evaluate(encode_board(board))


class MaterialEvaluator:
    """Fixed, non-evolving baseline: scores purely by material balance,
    normalized roughly into [-1, 1]. Used so fitness is anchored to
    something resembling real chess skill, not just relative to other
    evolving (possibly equally bad) genomes."""

    def score(self, board: chess.Board) -> float:
        return material_balance(board) / 39.0


# ─── Move selection ───────────────────────────────────────────────────────────

@torch.no_grad()
def choose_move(evaluator, board: chess.Board) -> chess.Move:
    """
    1-ply lookahead: try every legal move, score the resulting position
    with the evaluator, and pick the best one for the side to move.
    White wants a high score, Black wants a low score.
    `evaluator` must have a .score(board) -> float method.
    """
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = None
    white_to_move = board.turn == chess.WHITE

    for move in legal_moves:
        board.push(move)

        if board.is_checkmate():
            # Immediate mate is maximum reward for the mover (and maximum pain for the defender).
            score = 1.0 if white_to_move else -1.0
        else:
            score = evaluator.score(board)

        board.pop()

        if best_score is None:
            best_score, best_move = score, move
        elif white_to_move and score > best_score:
            best_score, best_move = score, move
        elif (not white_to_move) and score < best_score:
            best_score, best_move = score, move

    return best_move


# ─── Full game simulation ─────────────────────────────────────────────────────

def play_game(evaluator_white, evaluator_black, max_moves: int = 150) -> dict:
    """
    Plays a full game between two evaluators (each with a .score(board)
    method — see NetEvaluator / MaterialEvaluator).
    Returns a dict: {"result": "1-0"/"0-1"/"1/2-1/2", "moves": int,
                      "material_diff": float}
    """
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        evaluator = evaluator_white if board.turn == chess.WHITE else evaluator_black
        move = choose_move(evaluator, board)
        if move is None:
            break
        board.push(move)
        move_count += 1

    if board.is_checkmate():
        result = "0-1" if board.turn == chess.WHITE else "1-0"
    elif board.is_game_over():
        result = "1/2-1/2"
    else:
        # Hit move cap without a decisive result -> score by material.
        result = "1/2-1/2"

    return {
        "result": result,
        "moves": move_count,
        "material_diff": material_balance(board),
    }
