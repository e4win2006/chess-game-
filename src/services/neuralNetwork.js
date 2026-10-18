/**
 * neuralNetwork.js
 *
 * JavaScript port of the Python `nural network/` module + online learning.
 *
 * New in this version:
 *   • Manual backpropagation (backward()) so the network can learn in-browser
 *   • trainOnPosition(fen, target, lr) — train one position toward a target score
 *   • learnFromMove(fenBefore, fenAfter, lr) — TD(0) temporal-difference update
 *   • learnFromGameResult(positionHistory, result) — end-of-game propagation
 *   • getLearningStats() — { totalUpdates, recentLoss, lossHistory }
 *
 * Architecture: 773 → 128 → 64 → 32 → 1  (ReLU hidden, Tanh output)
 * Score ∈ [-1, 1]:  +1 = great for White,  -1 = great for Black
 */

import { Chess } from 'chess.js';

// ─── Constants ────────────────────────────────────────────────────────────────

const INPUT_SIZE   = 773;
const HIDDEN_SIZES = [128, 64, 32];
const PIECE_ORDER  = ['p', 'n', 'b', 'r', 'q', 'k'];
const PIECE_VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

// ─── Xavier weight init ───────────────────────────────────────────────────────

function xavierUniform(fanIn, fanOut) {
  const limit = Math.sqrt(6.0 / (fanIn + fanOut));
  return () => (Math.random() * 2 - 1) * limit;
}

function createLayer(inSize, outSize) {
  const init    = xavierUniform(inSize, outSize);
  const weights = new Float32Array(inSize * outSize);
  const bias    = new Float32Array(outSize);
  for (let i = 0; i < weights.length; i++) weights[i] = init();
  return { weights, bias, inSize, outSize };
}

// ─── ChessEvalNet ─────────────────────────────────────────────────────────────

class ChessEvalNet {
  constructor() {
    const sizes = [INPUT_SIZE, ...HIDDEN_SIZES, 1];
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      this.layers.push(createLayer(sizes[i], sizes[i + 1]));
    }
  }

  /** Forward pass — returns scalar output ∈ [-1, 1] */
  forward(x) {
    let cur = x;
    for (let li = 0; li < this.layers.length; li++) {
      const { weights, bias, inSize, outSize } = this.layers[li];
      const out  = new Float32Array(outSize);
      const last = li === this.layers.length - 1;
      for (let j = 0; j < outSize; j++) {
        let sum = bias[j];
        const off = j * inSize;
        for (let i = 0; i < inSize; i++) sum += weights[off + i] * cur[i];
        out[j] = last ? Math.tanh(sum) : Math.max(0, sum); // Tanh or ReLU
      }
      cur = out;
    }
    return cur[0];
  }

  evaluate(vec) { return this.forward(vec); }

  /**
   * Backpropagation — standard SGD weight update.
   * @param {Float32Array} featureVec  Input (773-dim)
   * @param {number}       target      Desired output ∈ [-1, 1]
   * @param {number}       lr          Learning rate
   * @returns {number}                 Squared loss before the update
   */
  backward(featureVec, target, lr = 0.001) {
    // ── Forward pass, storing intermediates ───────────────────────────────
    const pres = [];             // pre-activation at each layer
    const acts = [featureVec];   // activations including input

    let cur = featureVec;
    for (let li = 0; li < this.layers.length; li++) {
      const { weights, bias, inSize, outSize } = this.layers[li];
      const pre  = new Float32Array(outSize);
      const act  = new Float32Array(outSize);
      const last = li === this.layers.length - 1;
      for (let j = 0; j < outSize; j++) {
        let sum = bias[j];
        const off = j * inSize;
        for (let i = 0; i < inSize; i++) sum += weights[off + i] * cur[i];
        pre[j] = sum;
        act[j] = last ? Math.tanh(sum) : Math.max(0, sum);
      }
      pres.push(pre);
      acts.push(act);
      cur = act;
    }

    const output = cur[0];
    const loss   = 0.5 * (output - target) ** 2;

    // ── Backward pass ─────────────────────────────────────────────────────
    // Output layer: d(tanh(z)) = 1 - tanh²(z) = 1 - output²
    let delta = new Float32Array([( output - target) * (1 - output * output)]);

    for (let li = this.layers.length - 1; li >= 0; li--) {
      const { weights, bias, inSize, outSize } = this.layers[li];
      const prevActs = acts[li]; // activations flowing INTO this layer

      // Propagate delta to previous layer (before weight update!)
      let prevDelta = null;
      if (li > 0) {
        prevDelta       = new Float32Array(inSize);
        const prevPre   = pres[li - 1];
        for (let i = 0; i < inSize; i++) {
          let s = 0;
          for (let j = 0; j < outSize; j++) s += weights[j * inSize + i] * delta[j];
          prevDelta[i] = prevPre[i] > 0 ? s : 0; // ReLU gradient
        }
      }

      // Update weights and biases (with gradient clipping ±1)
      for (let j = 0; j < outSize; j++) {
        const dj  = Math.max(-1, Math.min(1, delta[j])); // clip
        const off = j * inSize;
        for (let i = 0; i < inSize; i++) {
          weights[off + i] -= lr * dj * prevActs[i];
        }
        bias[j] -= lr * dj;
      }

      delta = prevDelta;
    }

    return loss;
  }

  saveWeights() {
    return {
      version: 1,
      layers: this.layers.map(l => ({
        weights: Array.from(l.weights),
        bias: Array.from(l.bias),
      })),
    };
  }

  loadWeights(obj) {
    if (!obj?.layers) { console.warn('[NeuralNet] Invalid weights'); return; }
    obj.layers.forEach((l, i) => {
      if (i >= this.layers.length) return;
      this.layers[i].weights = new Float32Array(l.weights);
      this.layers[i].bias    = new Float32Array(l.bias);
    });
    console.info('[NeuralNet] Weights loaded ✓');
  }
}

// Singleton network + learning stats
const _net = new ChessEvalNet();

const _stats = {
  totalUpdates : 0,
  recentLosses : [],  // last 50 losses
  get avgLoss() {
    if (!this.recentLosses.length) return null;
    return this.recentLosses.reduce((a, b) => a + b, 0) / this.recentLosses.length;
  },
  pushLoss(l) {
    this.recentLosses.push(l);
    if (this.recentLosses.length > 50) this.recentLosses.shift();
    this.totalUpdates++;
  }
};

const STORAGE_KEY = 'chess_arena_neural_weights';
const STATS_KEY   = 'chess_arena_neural_stats';

export function saveWeightsToStorage() {
  try {
    if (typeof localStorage === 'undefined') return;
    const data = _net.saveWeights();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    localStorage.setItem(STATS_KEY, JSON.stringify({
      totalUpdates: _stats.totalUpdates,
      recentLosses: _stats.recentLosses.slice(-20),
    }));
  } catch {
    // localStorage may be disabled
  }
}

export function loadWeightsFromStorage() {
  try {
    if (typeof localStorage === 'undefined') return;
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      _net.loadWeights(JSON.parse(raw));
    }
    const rawStats = localStorage.getItem(STATS_KEY);
    if (rawStats) {
      const s = JSON.parse(rawStats);
      if (typeof s.totalUpdates === 'number') _stats.totalUpdates = s.totalUpdates;
      if (Array.isArray(s.recentLosses)) _stats.recentLosses = s.recentLosses;
    }
  } catch (e) {
    console.warn('[NeuralNet] Could not load stored weights:', e.message);
  }
}

// Initialize weights from local storage if previously trained
loadWeightsFromStorage();

// ─── Board encoder — mirrors chess_env.py encode_board() ─────────────────────

function encodeBoard(chess) {
  const vec   = new Float32Array(INPUT_SIZE);
  const board = chess.board();

  for (let rank = 0; rank < 8; rank++) {
    for (let file = 0; file < 8; file++) {
      const piece = board[rank][file];
      if (!piece) continue;
      const idx         = PIECE_ORDER.indexOf(piece.type);
      if (idx < 0) continue;
      const colorOffset = piece.color === 'w' ? 0 : 6;
      const plane       = colorOffset + idx;
      const chessRank   = 7 - rank; // chess.js top-row = rank 8
      vec[plane * 64 + chessRank * 8 + file] = 1.0;
    }
  }

  const base     = 768;
  const castling = chess.fen().split(' ')[2];
  vec[base + 0]  = chess.turn() === 'w' ? 1.0 : 0.0;
  vec[base + 1]  = castling.includes('K') ? 1.0 : 0.0;
  vec[base + 2]  = castling.includes('Q') ? 1.0 : 0.0;
  vec[base + 3]  = castling.includes('k') ? 1.0 : 0.0;
  vec[base + 4]  = castling.includes('q') ? 1.0 : 0.0;
  return vec;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function materialBalance(chess) {
  let total = 0;
  for (const row of chess.board()) {
    for (const p of row) {
      if (!p) continue;
      total += (PIECE_VALUES[p.type] ?? 0) * (p.color === 'w' ? 1 : -1);
    }
  }
  return total;
}

function scorePosition(chess) {
  try   { return _net.evaluate(encodeBoard(chess)); }
  catch { return materialBalance(chess) / 39.0; }
}

// ─── 1-ply lookahead (move selection) ────────────────────────────────────────

function chooseMoveNeural(chess, difficulty) {
  const legalMoves = chess.moves({ verbose: true });
  if (!legalMoves.length) return { bestSan: null, log: [] };

  const whiteToMove = chess.turn() === 'w';
  const log         = [];

  for (const move of legalMoves) {
    const scratch = new Chess(chess.fen());
    scratch.move(move);

    let score;
    let isCheckmate = false;
    if (scratch.isCheckmate()) {
      score       = whiteToMove ? 1.0 : -1.0;
      isCheckmate = true;
    } else {
      score = scorePosition(scratch);
      if (difficulty === 'easy') score += (Math.random() - 0.5) * 0.4;
    }

    log.push({ san: move.san, from: move.from, to: move.to,
               score, isCheckmate, isBest: false });
  }

  log.sort((a, b) => whiteToMove ? b.score - a.score : a.score - b.score);
  if (log.length) log[0].isBest = true;

  return { bestSan: log[0]?.san ?? null, log };
}

// ─── Reward & Penalty (Pain) Shaping ─────────────────────────────────────────

/**
 * Computes immediate reward (+) or pain (-) for a move from White's perspective.
 * Positive = favorable for White, Negative = favorable for Black.
 * Captures, checks, and mates grant rewards; blunders, lost material, and checks grant pain.
 */
function computeMoveReward(chessBefore, chessAfter) {
  const moverColor = chessBefore.turn(); // 'w' | 'b'
  const isWhite    = moverColor === 'w';

  // 1. Material changes: reward for captures, pain for losing pieces
  const matBefore = materialBalance(chessBefore);
  const matAfter  = materialBalance(chessAfter);
  const matDiff   = matAfter - matBefore; // + for White gain, - for Black gain

  // Reward: 0.06 per pawn point (pawn: +0.06, knight/bishop: +0.18, rook: +0.30, queen: +0.54)
  // Pain: equal penalty when material is lost
  let reward = matDiff * 0.06;

  // 2. Check bonus / pain: giving check grants reward, receiving check grants pain
  if (chessAfter.inCheck()) {
    reward += isWhite ? 0.08 : -0.08;
  }

  // 3. Decisive Checkmate: ultimate reward for winner, max pain for loser
  if (chessAfter.isCheckmate()) {
    reward += isWhite ? 1.0 : -1.0;
  }

  // 4. Center control reward (e4, d4, e5, d5)
  const centerSquares = ['e4', 'd4', 'e5', 'd5'];
  for (const sq of centerSquares) {
    const p = chessAfter.get(sq);
    if (p && p.color === moverColor) {
      reward += isWhite ? 0.015 : -0.015;
    }
  }

  return reward;
}

// ─── Online learning ──────────────────────────────────────────────────────────

/**
 * TD(0) update with reward & pain shaping (Bellman equation):
 * Target = clamp(Reward + gamma * V(s'), -1, 1)
 *
 * Rewards: piece capture, delivering check, checkmate, center control
 * Pain: losing pieces, receiving check, getting checkmated
 *
 * @param {string} fenBefore  FEN before the move
 * @param {string} fenAfter   FEN after the move
 * @param {number} lr         Learning rate (default 0.001)
 */
export function learnFromMove(fenBefore, fenAfter, lr = 0.001) {
  try {
    const chessBefore = new Chess(fenBefore);
    const chessAfter  = new Chess(fenAfter);

    // Compute immediate reward (+) or pain (-) for the move
    const reward    = computeMoveReward(chessBefore, chessAfter);
    const nextValue = scorePosition(chessAfter);

    // Bellman target with 0.95 discount factor: Target = R + gamma * V(s')
    const target    = Math.max(-1.0, Math.min(1.0, reward + 0.95 * nextValue));

    const vec       = encodeBoard(chessBefore);
    const loss      = _net.backward(vec, target, lr);
    _stats.pushLoss(loss);

    if (_stats.totalUpdates % 10 === 0) {
      saveWeightsToStorage();
    }
  } catch (e) {
    console.warn('[NeuralNet] learnFromMove error:', e.message);
  }
}

/**
 * Imitation learning: when another AI (Minimax / Heuristic) picks a move,
 * train the network to agree that the resulting position value is correct
 * for the POSITION BEFORE the move.
 *
 * @param {string} fenBefore   FEN before the move
 * @param {string} fenAfter    FEN after the chosen move
 * @param {string} playerType  'minimax' | 'heuristic' | string
 * @param {number} lr          Learning rate (default 0.001)
 */
export function learnFromAIMove(fenBefore, fenAfter, playerType, lr = 0.001) {
  learnFromMove(fenBefore, fenAfter, lr);

  try {
    const chessBefore = new Chess(fenBefore);
    const moverTurn = chessBefore.turn(); // 'w' | 'b'
    // Reinforce favorable outcome for the mover
    const target = moverTurn === 'w' ? 0.35 : -0.35;
    const chessAfter = new Chess(fenAfter);
    const loss = _net.backward(encodeBoard(chessAfter), target, lr * 0.6);
    _stats.pushLoss(loss);
  } catch {}
}

/**
 * End-of-game propagation: train all recorded positions toward the
 * game result (+1 White win / -1 Black win / 0 draw) with temporal discounting.
 *
 * @param {{ fen: string, turn: string }[]} history  All positions recorded during the game
 * @param {'white'|'black'|'draw'} result
 * @param {number} lr   Learning rate (default 0.0008)
 */
export function learnFromGameResult(history, result, lr = 0.0008) {
  if (!history || !history.length) return;

  const baseOutcome = result === 'white' ? 1.0 : (result === 'black' ? -1.0 : 0.0);
  const totalSteps = history.length;
  const gamma = 0.96; // Discount factor

  for (let i = 0; i < totalSteps; i++) {
    const { fen } = history[i];
    try {
      const chess = new Chess(fen);
      const stepsFromEnd = totalSteps - 1 - i;
      const target = baseOutcome * Math.pow(gamma, Math.min(stepsFromEnd, 25));
      const loss = _net.backward(encodeBoard(chess), target, lr);
      _stats.pushLoss(loss);
    } catch (e) {
      console.warn('[NeuralNet] learnFromGameResult error:', e.message);
    }
  }

  saveWeightsToStorage();
  console.info(`[NeuralNet] Learned from game outcome (${result}) across ${totalSteps} positions. Total steps: ${_stats.totalUpdates}`);
}

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Main move-picking entry point (called from App.jsx).
 * Returns: { move, thinking, candidateSquares, responseMs }
 */
export async function getNeuralNetMove(fen, legalMoves, difficulty, _color) {
  await new Promise(r => setTimeout(r, 0)); // yield to React

  const t0 = performance.now();

  try {
    const chess = new Chess(fen);
    const { bestSan, log } = chooseMoveNeural(chess, difficulty);

    const responseMs       = Math.round(performance.now() - t0);
    const candidateSquares = [...new Set(log.slice(0, 5).map(e => e.from))];
    const chosen           = bestSan && legalMoves.includes(bestSan) ? bestSan : null;

    if (chosen) {
      return { move: chosen, thinking: log, candidateSquares, responseMs };
    }
    console.warn('[NeuralNet] Move not in legalMoves, falling back:', bestSan);
  } catch (err) {
    console.error('[NeuralNet] Error:', err);
  }

  const responseMs = Math.round(performance.now() - t0);
  const fallback   = legalMoves[Math.floor(Math.random() * legalMoves.length)];
  return { move: fallback, thinking: [], candidateSquares: [], responseMs };
}

/** Load pre-trained weights from Python export */
export function loadWeights(obj) { _net.loadWeights(obj); }

/** Learning statistics for the UI */
export function getLearningStats() {
  return {
    totalUpdates : _stats.totalUpdates,
    avgLoss      : _stats.avgLoss,
    lossHistory  : [..._stats.recentLosses],
  };
}

/** Architecture description */
export const NET_INFO = `ChessEvalNet [${INPUT_SIZE}→${HIDDEN_SIZES.join('→')}→1 | ReLU+Tanh | Xavier-init | online TD learning]`;
