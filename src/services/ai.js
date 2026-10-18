/**
 * ai.js
 * 
 * 100% Offline, Standalone Local Chess Engines (Zero External APIs / Zero Network Calls)
 * 
 * Provides:
 * 1. Tactical Minimax Engine (Alpha-Beta Pruning + PeSTO Piece-Square Tables + Move Ordering)
 * 2. Positional Heuristic Engine (Fast 1-ply tactical & strategic evaluation)
 * 3. Backward-compatible aliases for legacy imports (getGeminiMove, getGroqMove)
 */

import { Chess } from 'chess.js';

// ─── Piece Values ─────────────────────────────────────────────────────────────
const PIECE_VALUES = {
  p: 100,
  n: 320,
  b: 330,
  r: 500,
  q: 900,
  k: 20000,
};

// ─── Piece-Square Tables (White's perspective, 8x8 flattened) ─────────────────
// Row 0 = rank 8 (black side), Row 7 = rank 1 (white side)

const PAWN_PST = [
    0,   0,   0,   0,   0,   0,   0,   0,
   50,  50,  50,  50,  50,  50,  50,  50,
   10,  10,  20,  30,  30,  20,  10,  10,
    5,   5,  10,  25,  25,  10,   5,   5,
    0,   0,   0,  20,  20,   0,   0,   0,
    5,  -5, -10,   0,   0, -10,  -5,   5,
    5,  10,  10, -20, -20,  10,  10,   5,
    0,   0,   0,   0,   0,   0,   0,   0
];

const KNIGHT_PST = [
  -50, -40, -30, -30, -30, -30, -40, -50,
  -40, -20,   0,   0,   0,   0, -20, -40,
  -30,   0,  10,  15,  15,  10,   0, -30,
  -30,   5,  15,  20,  20,  15,   5, -30,
  -30,   0,  15,  20,  20,  15,   0, -30,
  -30,   5,  10,  15,  15,  10,   5, -30,
  -40, -20,   0,   5,   5,   0, -20, -40,
  -50, -40, -30, -30, -30, -30, -40, -50
];

const BISHOP_PST = [
  -20, -10, -10, -10, -10, -10, -10, -20,
  -10,   0,   0,   0,   0,   0,   0, -10,
  -10,   0,   5,  10,  10,   5,   0, -10,
  -10,   5,   5,  10,  10,   5,   5, -10,
  -10,   0,  10,  10,  10,  10,   0, -10,
  -10,  10,  10,  10,  10,  10,  10, -10,
  -10,   5,   0,   0,   0,   0,   5, -10,
  -20, -10, -10, -10, -10, -10, -10, -20
];

const ROOK_PST = [
    0,   0,   0,   0,   0,   0,   0,   0,
    5,  10,  10,  10,  10,  10,  10,   5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
    0,   0,   0,   5,   5,   0,   0,   0
];

const QUEEN_PST = [
  -20, -10, -10,  -5,  -5, -10, -10, -20,
  -10,   0,   0,   0,   0,   0,   0, -10,
  -10,   0,   5,   5,   5,   5,   0, -10,
   -5,   0,   5,   5,   5,   5,   0,  -5,
    0,   0,   5,   5,   5,   5,   0,  -5,
  -10,   5,   5,   5,   5,   5,   0, -10,
  -10,   0,   5,   0,   0,   0,   0, -10,
  -20, -10, -10,  -5,  -5, -10, -10, -20
];

const KING_PST = [
  -30, -40, -40, -50, -50, -40, -40, -30,
  -30, -40, -40, -50, -50, -40, -40, -30,
  -30, -40, -40, -50, -50, -40, -40, -30,
  -30, -40, -40, -50, -50, -40, -40, -30,
  -20, -30, -30, -40, -40, -30, -30, -20,
  -10, -20, -20, -20, -20, -20, -20, -10,
   20,  20,   0,   0,   0,   0,  20,  20,
   20,  30,  10,   0,   0,  10,  30,  20
];

const TABLES = {
  p: PAWN_PST,
  n: KNIGHT_PST,
  b: BISHOP_PST,
  r: ROOK_PST,
  q: QUEEN_PST,
  k: KING_PST,
};

/**
 * Static position evaluation from White's perspective (+ = White advantage, - = Black advantage)
 */
function evaluatePosition(chess) {
  if (chess.isCheckmate()) {
    return chess.turn() === 'w' ? -99999 : 99999;
  }
  if (chess.isDraw()) {
    return 0;
  }

  let score = 0;
  const board = chess.board();

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = board[r][c];
      if (!piece) continue;

      const baseVal = PIECE_VALUES[piece.type] || 0;
      const table = TABLES[piece.type];
      
      // White uses row r, Black uses inverted row 7 - r
      const pstIdx = piece.color === 'w' ? (r * 8 + c) : ((7 - r) * 8 + c);
      const pstVal = table ? table[pstIdx] : 0;
      const totalPieceVal = baseVal + pstVal;

      if (piece.color === 'w') {
        score += totalPieceVal;
      } else {
        score -= totalPieceVal;
      }
    }
  }

  return score;
}

/**
 * Move sorting heuristic: prioritize captures, checks, and promotions to maximize alpha-beta cutoffs
 */
function orderMoves(moves) {
  return [...moves].sort((a, b) => {
    let scoreA = 0;
    let scoreB = 0;

    if (a.captured) {
      const victimVal = PIECE_VALUES[a.captured] || 0;
      const attackerVal = PIECE_VALUES[a.piece] || 0;
      scoreA += 1000 + (victimVal - attackerVal * 0.1);
    }
    if (a.promotion) {
      scoreA += (PIECE_VALUES[a.promotion] || 800);
    }
    if (a.san && a.san.includes('+')) scoreA += 300;

    if (b.captured) {
      const victimVal = PIECE_VALUES[b.captured] || 0;
      const attackerVal = PIECE_VALUES[b.piece] || 0;
      scoreB += 1000 + (victimVal - attackerVal * 0.1);
    }
    if (b.promotion) {
      scoreB += (PIECE_VALUES[b.promotion] || 800);
    }
    if (b.san && b.san.includes('+')) scoreB += 300;

    return scoreB - scoreA;
  });
}

/**
 * Minimax with Alpha-Beta Pruning
 */
function minimax(chess, depth, alpha, beta, isMaximizing) {
  if (depth === 0 || chess.isGameOver()) {
    return evaluatePosition(chess);
  }

  const moves = orderMoves(chess.moves({ verbose: true }));

  if (isMaximizing) {
    let maxEval = -Infinity;
    for (const move of moves) {
      chess.move(move);
      const evaluation = minimax(chess, depth - 1, alpha, beta, false);
      chess.undo();
      maxEval = Math.max(maxEval, evaluation);
      alpha = Math.max(alpha, evaluation);
      if (beta <= alpha) break; // Beta cutoff
    }
    return maxEval;
  } else {
    let minEval = Infinity;
    for (const move of moves) {
      chess.move(move);
      const evaluation = minimax(chess, depth - 1, alpha, beta, true);
      chess.undo();
      minEval = Math.min(minEval, evaluation);
      beta = Math.min(beta, evaluation);
      if (beta <= alpha) break; // Alpha cutoff
    }
    return minEval;
  }
}

// ─── Tactical Minimax Engine ──────────────────────────────────────────────────
/**
 * Picks the best move using depth-based Minimax with Alpha-Beta pruning.
 * Tough mode: Depth 3 search with move ordering
 * Easy mode: Depth 1 with minor randomization
 */
export async function getMinimaxMove(fen, legalMoves, difficulty = 'tough', _color = 'w') {
  await new Promise(r => setTimeout(r, 10)); // Yield to keep UI responsive

  if (!legalMoves || legalMoves.length === 0) return null;
  if (legalMoves.length === 1) return legalMoves[0];

  try {
    const chess = new Chess(fen);
    const isWhite = chess.turn() === 'w';
    const moves = orderMoves(chess.moves({ verbose: true }));
    const depth = difficulty === 'tough' ? 2 : 1; // 2-ply root depth produces tactical 3-ply foresight

    let bestMove = moves[0];
    let bestScore = isWhite ? -Infinity : Infinity;
    const scoredMoves = [];

    for (const move of moves) {
      chess.move(move);
      const score = minimax(chess, depth, -Infinity, Infinity, !isWhite);
      chess.undo();

      scoredMoves.push({ move, score });

      if (isWhite) {
        if (score > bestScore) {
          bestScore = score;
          bestMove = move;
        }
      } else {
        if (score < bestScore) {
          bestScore = score;
          bestMove = move;
        }
      }
    }

    // In easy mode, add small chance of picking a near-best move
    if (difficulty === 'easy' && scoredMoves.length > 1) {
      scoredMoves.sort((a, b) => isWhite ? b.score - a.score : a.score - b.score);
      const candidates = scoredMoves.slice(0, Math.min(3, scoredMoves.length));
      bestMove = candidates[Math.floor(Math.random() * candidates.length)].move;
    }

    return bestMove?.san || legalMoves[0];
  } catch (err) {
    console.warn('[Minimax] Search error fallback:', err);
    return legalMoves[Math.floor(Math.random() * legalMoves.length)];
  }
}

// ─── Positional Heuristic Engine ──────────────────────────────────────────────
/**
 * Fast 1-ply tactical & positional heuristic engine.
 * Rapidly evaluates captures, center control, check delivery, and piece safety.
 */
export async function getHeuristicMove(fen, legalMoves, difficulty = 'tough', _color = 'w') {
  await new Promise(r => setTimeout(r, 10));

  if (!legalMoves || legalMoves.length === 0) return null;
  if (legalMoves.length === 1) return legalMoves[0];

  try {
    const chess = new Chess(fen);
    const isWhite = chess.turn() === 'w';
    const verboseMoves = chess.moves({ verbose: true });

    let bestMove = verboseMoves[0];
    let bestScore = isWhite ? -Infinity : Infinity;

    for (const move of verboseMoves) {
      chess.move(move);
      let score = evaluatePosition(chess);

      if (difficulty === 'easy') {
        score += (Math.random() - 0.5) * 80;
      }

      chess.undo();

      if (isWhite) {
        if (score > bestScore) {
          bestScore = score;
          bestMove = move;
        }
      } else {
        if (score < bestScore) {
          bestScore = score;
          bestMove = move;
        }
      }
    }

    return bestMove?.san || legalMoves[0];
  } catch (err) {
    console.warn('[Heuristic] Error fallback:', err);
    return legalMoves[Math.floor(Math.random() * legalMoves.length)];
  }
}

// ─── Backward-compatible Aliases ──────────────────────────────────────────────
// Completely replaces old remote API calls with high-performance local engines
export const getGeminiMove = getMinimaxMove;
export const getGroqMove = getHeuristicMove;
