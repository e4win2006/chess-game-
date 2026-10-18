import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import Controls from './components/Controls';
import PlayerCard from './components/PlayerCard';
import NeuralNetPanel from './components/NeuralNetPanel';
import { getMinimaxMove, getHeuristicMove } from './services/ai';
import {
  getNeuralNetMove,
  learnFromMove,
  learnFromAIMove,
  learnFromGameResult,
  getLearningStats,
} from './services/neuralNetwork';
import { playMoveSound, playCaptureSound, playCheckSound, playGameOverSound, playAntigravitySound, toggleSound, isSoundMuted } from './utils/sound';
import './index.css';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getCapturedPieces(chess) {
  const board = chess.board().flat().filter(Boolean);
  const whiteStock = { p: 8, n: 2, b: 2, r: 2, q: 1 };
  const blackStock = { p: 8, n: 2, b: 2, r: 2, q: 1 };
  for (const piece of board) {
    if (piece.color === 'w') { if (whiteStock[piece.type] > 0) whiteStock[piece.type]--; }
    else                     { if (blackStock[piece.type]  > 0) blackStock[piece.type]--;  }
  }
  const whiteCaptured = [];
  for (const [type, count] of Object.entries(blackStock))
    for (let i = 0; i < count; i++) whiteCaptured.push(type);
  const blackCaptured = [];
  for (const [type, count] of Object.entries(whiteStock))
    for (let i = 0; i < count; i++) blackCaptured.push(type);
  return { whiteCaptured, blackCaptured };
}

function getKingSquare(chess, color) {
  const board = chess.board();
  for (let r = 0; r < 8; r++)
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p && p.type === 'k' && p.color === color)
        return `${String.fromCharCode(97 + c)}${8 - r}`;
    }
  return null;
}

/** Check if a side has mating material against a bare king */
function hasMatingMaterial(chess, color) {
  const pieces = chess.board().flat().filter(p => p && p.color === color);
  const queens = pieces.filter(p => p.type === 'q').length;
  const rooks = pieces.filter(p => p.type === 'r').length;
  const pawns = pieces.filter(p => p.type === 'p').length;
  const bishops = pieces.filter(p => p.type === 'b').length;
  const knights = pieces.filter(p => p.type === 'n').length;
  if (queens > 0 || rooks > 0 || pawns > 0) return true;
  if (bishops >= 2) return true;
  if (bishops >= 1 && knights >= 1) return true;
  if (knights >= 2) return true;
  return false;
}

/** Determine draw details based on FIDE Laws of Chess */
function getFideDrawDetails(chess, posCount) {
  if (chess.isStalemate()) {
    return {
      isDraw: true,
      reason: '½-½ Drawn by Stalemate: Player to move has no legal moves and is not in check (FIDE Law 9.6).',
      code: 'stalemate',
    };
  }
  if (posCount >= 5) {
    return {
      isDraw: true,
      reason: '½-½ Mandatory Draw by Fivefold Repetition: The identical position occurred 5 times (FIDE Law 9.6.1).',
      code: 'fivefold_repetition',
    };
  }
  if (posCount >= 3 || chess.isThreefoldRepetition()) {
    return {
      isDraw: true,
      reason: '½-½ Drawn by Threefold Repetition: The identical position occurred 3 times (FIDE Law 9.2).',
      code: 'threefold_repetition',
    };
  }
  const halfMoves = parseInt(chess.fen().split(' ')[4], 10) || 0;
  if (halfMoves >= 150) {
    return {
      isDraw: true,
      reason: '½-½ Mandatory Draw by 75-Move Rule: 75 consecutive full moves without a capture or pawn advance (FIDE Law 9.6.2).',
      code: '75_moves',
    };
  }
  if (halfMoves >= 100 || chess.isDrawByFiftyMoves()) {
    return {
      isDraw: true,
      reason: '½-½ Drawn by 50-Move Rule: 50 consecutive full moves without a capture or pawn advance (FIDE Law 9.3).',
      code: '50_moves',
    };
  }
  if (chess.isInsufficientMaterial()) {
    return {
      isDraw: true,
      reason: '½-½ Drawn by Insufficient Mating Material: Dead position where checkmate is impossible (FIDE Law 9.6).',
      code: 'insufficient_material',
    };
  }
  if (chess.isDraw()) {
    return {
      isDraw: true,
      reason: '½-½ Game drawn.',
      code: 'draw',
    };
  }
  return { isDraw: false, reason: '', code: '' };
}

/** Format seconds into MM:SS */
function formatClock(seconds) {
  if (seconds === null || seconds === undefined) return '--:--';
  const m = Math.floor(Math.max(0, seconds) / 60);
  const s = Math.max(0, seconds) % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

const PROMOTION_GLYPHS = {
  w: { q: '♕', r: '♖', b: '♗', n: '♘' },
  b: { q: '♛', r: '♜', b: '♝', n: '♞' },
};
const PROMOTION_NAMES = {
  q: 'Queen',
  r: 'Rook',
  b: 'Bishop',
  n: 'Knight',
};

/** Determine winner from game state */
function getResultFromGame(chess, drawDetails, timeExpired, resigned) {
  if (resigned) return resigned === 'w' ? 'black' : 'white';
  if (timeExpired) {
    const opponent = timeExpired === 'w' ? 'b' : 'w';
    if (!hasMatingMaterial(chess, opponent)) return 'draw';
    return opponent === 'w' ? 'white' : 'black';
  }
  if (chess.isCheckmate()) return chess.turn() === 'w' ? 'black' : 'white';
  if (drawDetails?.isDraw || chess.isDraw()) return 'draw';
  return null;
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  // ── Stable game ref (avoids stale-closure bugs in async callbacks) ──────
  const [game, _setGame] = useState(() => new Chess());
  const gameRef = useRef(game);
  const setGame = useCallback((g) => { gameRef.current = g; _setGame(g); }, []);

  // ── Match settings ───────────────────────────────────────────────────────
  const [whitePlayer, setWhitePlayer] = useState('neural');
  const [blackPlayer, setBlackPlayer] = useState('minimax');
  const [difficulty,  setDifficulty]  = useState('tough');
  const [isPaused,    setIsPaused]    = useState(false);
  const [moveDelay,   setMoveDelay]   = useState(2000);
  const [timeControl, setTimeControl] = useState('none'); // 'none' | '180' | '300' | '600'
  const [whiteTime,   setWhiteTime]   = useState(null);
  const [blackTime,   setBlackTime]   = useState(null);
  const [timeExpired, setTimeExpired] = useState(null); // null | 'w' | 'b'
  const [resigned,    setResigned]    = useState(null); // null | 'w' | 'b'
  const [showResignModal, setShowResignModal] = useState(false);
  const [pendingPromotion, setPendingPromotion] = useState(null); // null | { from, to, color, responseMs }
  const [positionCounts, setPositionCounts] = useState({});

  // ── UI state ─────────────────────────────────────────────────────────────
  const [isThinking,   setIsThinking]   = useState(false);
  const [moveHistory,  setMoveHistory]  = useState([]);
  const [lastMove,     setLastMove]     = useState(null);
  const [boardWidth,   setBoardWidth]   = useState(480);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [flipBoard,    setFlipBoard]    = useState(false);
  const [soundMuted,   setSoundMuted]   = useState(() => isSoundMuted());
  const [isAntigravity, setIsAntigravity] = useState(false);
  const typedBufferRef = useRef('');

  // ── Antigravity toggle ───────────────────────────────────────────────────
  const toggleAntigravity = useCallback(() => {
    setIsAntigravity(prev => {
      const next = !prev;
      if (next) playAntigravitySound();
      return next;
    });
  }, []);

  // ── Response times (per player side) ─────────────────────────────────────
  const [whiteResponseMs, setWhiteResponseMs] = useState(null);
  const [blackResponseMs, setBlackResponseMs] = useState(null);

  // ── Neural Net state ──────────────────────────────────────────────────────
  const [neuralThinking,    setNeuralThinking]    = useState([]);
  const [neuralLastMove,    setNeuralLastMove]     = useState(null);
  const [neuralCandidates,  setNeuralCandidates]  = useState([]);
  const [neuralScanning,    setNeuralScanning]    = useState(false);
  const [learningStats,     setLearningStats]     = useState({ totalUpdates: 0, avgLoss: null });

  // ── Sound toggle ─────────────────────────────────────────────────────────
  const handleToggleSound = useCallback(() => {
    const next = toggleSound();
    setSoundMuted(next);
  }, []);

  // ── Fullscreen toggle ────────────────────────────────────────────────────
  const toggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => {
      const next = !prev;
      if (next) {
        if (document.documentElement.requestFullscreen && !document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(() => {});
        }
      } else {
        if (document.exitFullscreen && document.fullscreenElement) {
          document.exitFullscreen().catch(() => {});
        }
      }
      return next;
    });
  }, []);

  // ── Keyboard shortcut (F to toggle fullscreen, Esc to exit) ──────────────
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isFullscreen) {
        toggleFullscreen();
      } else if ((e.key === 'f' || e.key === 'F') && !['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) {
        toggleFullscreen();
      }
    };
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement && isFullscreen) {
        setIsFullscreen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, [isFullscreen, toggleFullscreen]);

  // ── Easter egg: Typing 'antigravity' triggers zero-G mode ─────────────────
  useEffect(() => {
    const handleEasterEgg = (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
      typedBufferRef.current = (typedBufferRef.current + e.key.toLowerCase()).slice(-11);
      if (typedBufferRef.current === 'antigravity') {
        toggleAntigravity();
        typedBufferRef.current = '';
      }
    };
    window.addEventListener('keydown', handleEasterEgg);
    return () => window.removeEventListener('keydown', handleEasterEgg);
  }, [toggleAntigravity]);

  // ── Position history for end-of-game learning ────────────────────────────
  // Each entry: { fen: string, turn: 'w'|'b', playerType: string }
  const positionHistoryRef = useRef([]);

  const isProcessingRef   = useRef(false);
  const movesContainerRef = useRef(null);

  // ── Refresh learning stats every 2s when neural is active ────────────────
  useEffect(() => {
    const neuralActive = whitePlayer === 'neural' || blackPlayer === 'neural';
    if (!neuralActive) return;
    const id = setInterval(() => setLearningStats(getLearningStats()), 2000);
    return () => clearInterval(id);
  }, [whitePlayer, blackPlayer]);

  // ── Responsive board sizing (optimized for standard & fullscreen) ────────
  useEffect(() => {
    const upd = () => {
      const isMobile = window.innerWidth < 850;
      const vh = window.innerHeight;
      const vw = window.innerWidth;
      if (isFullscreen) {
        // Subtract vertical footprint of top bar, player cards, floating dock, and margins (~210px)
        const maxH = vh - (isMobile ? 180 : 210);
        const maxW = vw - (isMobile ? 16 : 32);
        const size = Math.floor(Math.max(200, Math.min(maxW, maxH, 840)));
        setBoardWidth(size);
      } else {
        const maxW = isMobile ? Math.min(vw - 36, 420)
                              : Math.min(vw * 0.44, 460);
        const maxH = vh - 300;
        setBoardWidth(Math.floor(Math.max(260, Math.min(maxW, maxH > 280 ? maxH : 420))));
      }
    };
    upd();
    window.addEventListener('resize', upd);
    return () => window.removeEventListener('resize', upd);
  }, [isFullscreen]);

  // ── Auto-scroll move history ──────────────────────────────────────────────
  useEffect(() => {
    if (movesContainerRef.current)
      movesContainerRef.current.scrollTop = movesContainerRef.current.scrollHeight;
  }, [moveHistory]);

  // ── Clock timeout handler ──────────────────────────────────────────────
  const handleTimeout = useCallback((flaggedTurn) => {
    setTimeExpired(flaggedTurn);
    playGameOverSound();
    const opponentTurn = flaggedTurn === 'w' ? 'b' : 'w';
    const opponentHasMaterial = hasMatingMaterial(gameRef.current, opponentTurn);

    if (!opponentHasMaterial) {
      // FIDE Law 6.9: Draw if opponent cannot checkmate by any series of legal moves
      learnFromGameResult(positionHistoryRef.current, 'draw', 0.0003);
    } else {
      learnFromGameResult(positionHistoryRef.current, opponentTurn === 'w' ? 'white' : 'black', 0.0003);
    }
    setLearningStats(getLearningStats());
  }, []);

  // ── Clock timer tick ──────────────────────────────────────────────────────
  useEffect(() => {
    if (timeControl === 'none' || isPaused || game.isGameOver() || timeExpired || resigned) return;
    if (moveHistory.length === 0) return;

    const timerId = setInterval(() => {
      const turn = gameRef.current.turn(); // 'w' | 'b'
      if (turn === 'w') {
        setWhiteTime(prev => {
          if (prev === null) return null;
          if (prev <= 1) {
            clearInterval(timerId);
            handleTimeout('w');
            return 0;
          }
          return prev - 1;
        });
      } else {
        setBlackTime(prev => {
          if (prev === null) return null;
          if (prev <= 1) {
            clearInterval(timerId);
            handleTimeout('b');
            return 0;
          }
          return prev - 1;
        });
      }
    }, 1000);

    return () => clearInterval(timerId);
  }, [timeControl, isPaused, game, timeExpired, resigned, moveHistory.length, handleTimeout]);

  // ── Time control selector handler ────────────────────────────────────────
  const handleSetTimeControl = useCallback((newTc) => {
    setTimeControl(newTc);
    if (newTc === 'none') {
      setWhiteTime(null);
      setBlackTime(null);
    } else {
      const s = parseInt(newTc, 10);
      setWhiteTime(s);
      setBlackTime(s);
    }
  }, []);

  // ── Resignation confirmation handler ─────────────────────────────────────
  const confirmResign = useCallback(() => {
    const activeTurn = gameRef.current.turn();
    let color = activeTurn;
    if (whitePlayer === 'manual' && blackPlayer !== 'manual') color = 'w';
    else if (blackPlayer === 'manual' && whitePlayer !== 'manual') color = 'b';

    setResigned(color);
    setShowResignModal(false);
    playGameOverSound();

    learnFromGameResult(positionHistoryRef.current, color === 'w' ? 'black' : 'white', 0.0003);
    setLearningStats(getLearningStats());
  }, [whitePlayer, blackPlayer]);

  // ─── makeMove ─────────────────────────────────────────────────────────────
  // Always reads gameRef.current — never a stale closure.
  // Also feeds every move into the neural net for online learning.
  const makeMove = useCallback((moveParam, playerType = 'unknown', responseMs = null) => {
    try {
      const fenBefore  = gameRef.current.fen();
      const turn       = gameRef.current.turn(); // 'w' | 'b'

      const freshGame  = new Chess(fenBefore);
      const moveResult = freshGame.move(moveParam);
      if (!moveResult) return false;

      const fenAfter = freshGame.fen();

      // Record position for end-of-game learning
      positionHistoryRef.current.push({ fen: fenBefore, turn, playerType });

      // Track FIDE position count for threefold / fivefold repetition
      const posKey = freshGame.fen().split(' ').slice(0, 4).join(' ');
      let posCount = 1;
      setPositionCounts(prev => {
        const nextCount = (prev[posKey] || 0) + 1;
        posCount = nextCount;
        return { ...prev, [posKey]: nextCount };
      });
      const drawDetails = getFideDrawDetails(freshGame, posCount);

      if (playerType === 'neural') {
        // Net's own move: TD consistency update
        learnFromMove(fenBefore, fenAfter, 0.001);
      } else if (['minimax', 'heuristic'].includes(playerType)) {
        // Imitation learning from offline AI engines
        learnFromAIMove(fenBefore, fenAfter, playerType, 0.001);
      } else if (playerType === 'manual') {
        // Human moves: the network learns from player decisions
        learnFromMove(fenBefore, fenAfter, 0.0008);
      } else {
        // Any other move played by anyone
        learnFromMove(fenBefore, fenAfter, 0.0005);
      }
      setLearningStats(getLearningStats());

      // ── Update React state ─────────────────────────────────────────────
      setGame(freshGame);
      setLastMove({ from: moveResult.from, to: moveResult.to, san: moveResult.san });
      setMoveHistory(prev => [...prev, moveResult.san]);

      // ── Response time ─────────────────────────────────────────────────
      if (responseMs !== null) {
        if (turn === 'w') setWhiteResponseMs(responseMs);
        else              setBlackResponseMs(responseMs);
      }

      // ── Sound ─────────────────────────────────────────────────────────
      if (freshGame.isGameOver() || drawDetails.isDraw) playGameOverSound();
      else if (freshGame.inCheck()) playCheckSound();
      else if (moveResult.captured) playCaptureSound();
      else                          playMoveSound();

      // ── End-of-game learning trigger ──────────────────────────────────
      if (freshGame.isGameOver() || drawDetails.isDraw) {
        const result = getResultFromGame(freshGame, drawDetails, null, null);
        if (result) {
          setTimeout(() => {
            learnFromGameResult(positionHistoryRef.current, result, 0.0003);
            setLearningStats(getLearningStats());
          }, 100);
        }
      }

      return true;
    } catch (e) {
      console.warn('[makeMove] Invalid move:', moveParam, e.message);
      return false;
    }
  }, [setGame]);

  // ─── onDrop (human) ───────────────────────────────────────────────────────
  const onDrop = useCallback((src, tgt) => {
    if (isThinking || gameRef.current.isGameOver() || timeExpired || resigned) return false;
    const turn          = gameRef.current.turn() === 'w' ? 'white' : 'black';
    const currentPlayer = turn === 'white' ? whitePlayer : blackPlayer;
    if (currentPlayer !== 'manual') return false;

    const t0 = performance.now();
    const verboseMoves = gameRef.current.moves({ verbose: true });
    const isPromo = verboseMoves.some(m => m.from === src && m.to === tgt && m.promotion);

    if (isPromo) {
      setPendingPromotion({
        from: src,
        to: tgt,
        color: gameRef.current.turn(),
        responseMs: Math.round(performance.now() - t0),
      });
      return false;
    }

    const ok = makeMove({ from: src, to: tgt }, 'manual', Math.round(performance.now() - t0));
    return ok;
  }, [isThinking, whitePlayer, blackPlayer, timeExpired, resigned, makeMove]);

  // ─── Handle selection from promotion modal ────────────────────────────────
  const handleSelectPromotion = useCallback((promoPiece) => {
    if (!pendingPromotion) return;
    const { from, to, responseMs } = pendingPromotion;
    setPendingPromotion(null);
    makeMove({ from, to, promotion: promoPiece }, 'manual', responseMs);
  }, [pendingPromotion, makeMove]);

  // ─── executeAIMove ────────────────────────────────────────────────────────
  const executeAIMove = useCallback(async () => {
    if (gameRef.current.isGameOver() || isProcessingRef.current || timeExpired || resigned) return;

    const turn          = gameRef.current.turn() === 'w' ? 'white' : 'black';
    const currentPlayer = turn === 'white' ? whitePlayer : blackPlayer;
    if (currentPlayer === 'manual') return;

    isProcessingRef.current = true;
    setIsThinking(true);

    const fen        = gameRef.current.fen();
    const legalMoves = gameRef.current.moves();

    if (currentPlayer === 'neural') {
      setNeuralScanning(true);
      setNeuralThinking([]);
      setNeuralCandidates([]);
    }

    try {
      let chosenMove  = null;
      let responseMs  = null;

      if (currentPlayer === 'minimax') {
        const t0 = performance.now();
        chosenMove  = await getMinimaxMove(fen, legalMoves, difficulty, turn);
        responseMs  = Math.round(performance.now() - t0);

      } else if (currentPlayer === 'heuristic') {
        const t0 = performance.now();
        chosenMove  = await getHeuristicMove(fen, legalMoves, difficulty, turn);
        responseMs  = Math.round(performance.now() - t0);

      } else if (currentPlayer === 'neural') {
        const result = await getNeuralNetMove(fen, legalMoves, difficulty, turn);
        chosenMove   = result.move;
        responseMs   = result.responseMs;

        // Flash candidate squares for 500ms
        setNeuralCandidates(result.candidateSquares || []);
        await new Promise(r => setTimeout(r, 500));
        setNeuralCandidates([]);

        setNeuralThinking(result.thinking || []);
        setNeuralLastMove(result.move);
      }

      if (chosenMove) {
        makeMove(chosenMove, currentPlayer, responseMs);
      }

    } catch (err) {
      console.error('[executeAIMove]', err);
      const fallback = legalMoves[Math.floor(Math.random() * legalMoves.length)];
      if (fallback) makeMove(fallback, currentPlayer, null);
    } finally {
      isProcessingRef.current = false;
      setIsThinking(false);
      setNeuralScanning(false);
      setNeuralCandidates([]);
    }
  }, [whitePlayer, blackPlayer, difficulty, timeExpired, resigned, makeMove]);

  const stepAIMove = useCallback(() => {
    if (!isProcessingRef.current) executeAIMove();
  }, [executeAIMove]);

  // ─── AI Turn Execution Effect ─────────────────────────────────────────────
  useEffect(() => {
    if (game.isGameOver() || timeExpired || resigned) return;
    const posKey = game.fen().split(' ').slice(0, 4).join(' ');
    const posCount = positionCounts[posKey] || 1;
    const drawDetails = getFideDrawDetails(game, posCount);
    if (drawDetails.isDraw) return;

    const turn          = game.turn() === 'w' ? 'white' : 'black';
    const currentPlayer = turn === 'white' ? whitePlayer : blackPlayer;

    if (currentPlayer !== 'manual' && !isPaused) {
      const t = setTimeout(executeAIMove, moveDelay);
      return () => clearTimeout(t);
    }
  }, [game, whitePlayer, blackPlayer, isPaused, moveDelay, timeExpired, resigned, positionCounts, executeAIMove]);

  // ─── Reset ────────────────────────────────────────────────────────────────
  const onReset = useCallback(() => {
    const fresh = new Chess();
    setGame(fresh);
    setMoveHistory([]);
    setLastMove(null);
    setIsThinking(false);
    isProcessingRef.current = false;
    positionHistoryRef.current = [];
    setPositionCounts({});
    setTimeExpired(null);
    setResigned(null);
    setShowResignModal(false);
    setPendingPromotion(null);
    if (timeControl === 'none') {
      setWhiteTime(null);
      setBlackTime(null);
    } else {
      const s = parseInt(timeControl, 10);
      setWhiteTime(s);
      setBlackTime(s);
    }
    setWhiteResponseMs(null);
    setBlackResponseMs(null);
    setNeuralThinking([]);
    setNeuralLastMove(null);
    setNeuralCandidates([]);
    setNeuralScanning(false);
  }, [setGame, timeControl]);

  // ─── Derived values ───────────────────────────────────────────────────────
  const { whiteCaptured, blackCaptured } = getCapturedPieces(game);
  const isWhiteTurn       = game.turn() === 'w';
  const inCheck           = game.inCheck();
  const checkedKingSq     = inCheck ? getKingSquare(game, game.turn()) : null;
  const neuralIsActive    = whitePlayer === 'neural' || blackPlayer === 'neural';
  const currentTurnPlayer = game.turn() === 'w' ? whitePlayer : blackPlayer;
  const naturalOrientation = whitePlayer !== 'manual' && blackPlayer === 'manual' ? 'black' : 'white';
  const boardOrientation   = flipBoard ? (naturalOrientation === 'white' ? 'black' : 'white') : naturalOrientation;

  // ─── Derived status text (always in sync with current state) ──────────────
  let statusText = 'White to move';
  if (resigned) {
    const winner = resigned === 'w' ? 'Black' : 'White';
    statusText = `🏳️ ${resigned === 'w' ? 'White' : 'Black'} resigned. 🏆 ${winner} wins!`;
  } else if (timeExpired) {
    const opponent = timeExpired === 'w' ? 'b' : 'w';
    const opponentHasMaterial = hasMatingMaterial(game, opponent);
    if (!opponentHasMaterial) {
      statusText = '½-½ Drawn on time: Opponent has insufficient mating material (FIDE Law 6.9).';
    } else {
      const winner = opponent === 'w' ? 'White' : 'Black';
      statusText = `⏱️ ${timeExpired === 'w' ? 'White' : 'Black'} ran out of time! 🏆 ${winner} wins on time.`;
    }
  } else {
    const posKey = game.fen().split(' ').slice(0, 4).join(' ');
    const posCount = positionCounts[posKey] || 1;
    const drawDetails = getFideDrawDetails(game, posCount);

    if (game.isCheckmate()) {
      const winner = game.turn() === 'w' ? 'Black' : 'White';
      statusText = `Checkmate! 🏆 ${winner} wins.`;
    } else if (drawDetails.isDraw) {
      statusText = drawDetails.reason;
    } else if (game.isGameOver()) {
      statusText = 'Game Over.';
    } else {
      const turn          = game.turn() === 'w' ? 'white' : 'black';
      const currentPlayer = turn === 'white' ? whitePlayer : blackPlayer;
      const playerLabel   = {
        manual: 'Human',
        neural: 'Neural Net',
        minimax: 'Minimax AI',
        heuristic: 'Heuristic AI',
      }[currentPlayer] ?? currentPlayer;
      statusText = `${turn === 'white' ? 'White' : 'Black'} to move (${playerLabel})`;
    }
  }

  // ─── Square highlight styles ──────────────────────────────────────────────
  const customSquareStyles = {};

  if (lastMove) {
    customSquareStyles[lastMove.from] = { backgroundColor: 'rgba(255, 255, 51, 0.45)', boxShadow: 'inset 0 0 8px rgba(255, 255, 51, 0.5)' };
    customSquareStyles[lastMove.to]   = { backgroundColor: 'rgba(255, 255, 51, 0.45)', boxShadow: 'inset 0 0 10px rgba(255, 255, 51, 0.65)' };
  }
  if (checkedKingSq) {
    customSquareStyles[checkedKingSq] = {
      background: 'radial-gradient(circle, rgba(239, 68, 68, 0.85) 0%, rgba(239, 68, 68, 0) 70%)',
      borderRadius: '50%',
    };
  }
  if (currentTurnPlayer === 'neural' && neuralCandidates.length > 0) {
    neuralCandidates.forEach((sq, i) => {
      const alpha = Math.max(0.3, 0.65 - i * 0.08);
      customSquareStyles[sq] = {
        background: `radial-gradient(circle, rgba(16,185,129,${alpha}) 0%, rgba(6,182,212,${alpha * 0.55}) 55%, transparent 82%)`,
        boxShadow: `inset 0 0 16px rgba(16,185,129,${alpha})`,
        borderRadius: '50%',
        animation: 'neuralPulse 0.65s ease-in-out infinite alternate',
      };
    });
  }

  // Response time labels — assign based on orientation
  const topResponseMs    = boardOrientation === 'white' ? blackResponseMs : whiteResponseMs;
  const bottomResponseMs = boardOrientation === 'white' ? whiteResponseMs : blackResponseMs;
  const topPlayer        = boardOrientation === 'white' ? blackPlayer : whitePlayer;
  const bottomPlayer     = boardOrientation === 'white' ? whitePlayer : blackPlayer;
  const topIsTurn        = boardOrientation === 'white' ? !isWhiteTurn : isWhiteTurn;
  const topCaptured      = boardOrientation === 'white' ? blackCaptured : whiteCaptured;
  const bottomCaptured   = boardOrientation === 'white' ? whiteCaptured : blackCaptured;

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className={`app-container ${isFullscreen ? 'fullscreen-mode' : ''} ${isAntigravity ? 'antigravity-active' : ''}`}>
      {/* ── Antigravity Floating HUD Banner ── */}
      {isAntigravity && (
        <div className="antigravity-hud-banner glass-panel">
          <div className="ag-banner-left">
            <span className="ag-pulse-orb" />
            <div className="ag-banner-text">
              <span className="ag-title">🚀 Antigravity Zero-G Physics Active</span>
              <span className="ag-sub">Chessboard &amp; pieces are floating in zero gravity</span>
            </div>
          </div>
          <button id="restore-gravity-btn" className="restore-gravity-btn" onClick={() => setIsAntigravity(false)}>
            🌍 Restore Gravity
          </button>
        </div>
      )}

      {/* ── Top Bar (Fullscreen or Normal Header) ── */}
      {isFullscreen ? (
        <header className="fs-top-bar">
          <div className="fs-top-left">
            <span className="fs-brand">♛ Chess Arena</span>
            <div className={`status-badge ${game.isGameOver() || timeExpired || resigned ? 'game-over' : ''} ${neuralScanning ? 'neural-scanning-badge-header' : ''}`}>
              {neuralScanning ? '🧠 Scanning…' : statusText}
            </div>
            {lastMove && (
              <span className="fs-last-move-pill">
                Last: <strong>{lastMove.san}</strong>
              </span>
            )}
            {timeControl !== 'none' && (
              <span className="fs-clock-pill">
                ⏱ W: <strong>{formatClock(whiteTime)}</strong> · B: <strong>{formatClock(blackTime)}</strong>
              </span>
            )}
            {isAntigravity && (
              <span className="fs-ag-pill">🚀 Zero-G</span>
            )}
          </div>
          <div className="fs-top-right">
            <span className="fs-move-count-badge">{moveHistory.length} moves</span>
            <button
              id="exit-fullscreen-btn"
              className="fs-exit-btn"
              onClick={toggleFullscreen}
              title="Exit Fullscreen (Esc or F key)"
            >
              ✕ Exit Fullscreen <kbd>Esc</kbd>
            </button>
          </div>
        </header>
      ) : (
        <header className="title-section">
          <div className="title-row">
            <h1>AI Chess Arena</h1>
            <div className="header-actions">
              <button 
                id="header-antigravity-btn"
                className={`header-action-btn ${isAntigravity ? 'ag-active' : ''}`}
                onClick={toggleAntigravity}
                title="Toggle Antigravity zero-G physics (Easter egg: type 'antigravity')"
              >
                {isAntigravity ? '🌍 Gravity' : '🚀 Antigravity'}
              </button>
              <button 
                id="header-fullscreen-btn"
                className="header-action-btn header-fullscreen-btn" 
                onClick={toggleFullscreen}
                title="Expand chessboard to fullscreen view (F key)"
              >
                ⛶ Fullscreen Board
              </button>
            </div>
          </div>
          <div className={`status-badge ${game.isGameOver() || timeExpired || resigned ? 'game-over' : ''} ${neuralScanning ? 'neural-scanning-badge-header' : ''}`}>
            {neuralScanning ? '🧠 Neural Net scanning…' : statusText}
          </div>
          {neuralIsActive && learningStats.totalUpdates > 0 && (
            <div className="learning-stats-badge">
              🎓 {learningStats.totalUpdates} training steps
              {learningStats.avgLoss !== null && (
                <span className="learning-loss"> · loss {learningStats.avgLoss.toFixed(4)}</span>
              )}
            </div>
          )}
        </header>
      )}

      {/* ── Semantic Main: Board + Controls Layout ── */}
      <main className={`game-layout ${isFullscreen ? 'fs-game-layout' : ''}`}>
        {/* ── Board + Player Cards ───────────────────────────────── */}
        <section className={`board-section ${isFullscreen ? 'fs-board-section' : ''}`}>
          <PlayerCard
            side={boardOrientation === 'white' ? 'black' : 'white'}
            playerType={topPlayer}
            isTurn={topIsTurn}
            isThinking={isThinking}
            capturedPieces={topCaptured}
            responseMs={topResponseMs}
            timeRemaining={boardOrientation === 'white' ? blackTime : whiteTime}
            onResign={topPlayer === 'manual' ? () => setShowResignModal(true) : null}
          />

          <div 
            className={`board-wrapper glass-panel ${isFullscreen ? 'fs-board-wrapper' : ''}`} 
            style={{ 
              width: boardWidth + (isFullscreen ? 8 : 20), 
              height: boardWidth + (isFullscreen ? 8 : 20),
              touchAction: 'none',
            }}
          >
            <Chessboard
              options={{
                position: game.fen(),
                boardOrientation,
                animationDurationInMs: 320,
                showAnimations: true,
                squareStyles: customSquareStyles,
                arrows: lastMove ? [
                  {
                    startSquare: lastMove.from,
                    endSquare: lastMove.to,
                    color: 'rgba(234, 179, 8, 0.75)',
                  }
                ] : [],
                darkSquareStyle: { backgroundColor: '#739552' },
                lightSquareStyle: { backgroundColor: '#ebecd0' },
                darkSquareNotationStyle: { color: '#ebecd0', fontSize: '11px', fontWeight: '700' },
                lightSquareNotationStyle: { color: '#739552', fontSize: '11px', fontWeight: '700' },
                boardStyle: {
                  width: `${boardWidth}px`,
                  height: `${boardWidth}px`,
                  borderRadius: '10px',
                  touchAction: 'none',
                  boxShadow: isFullscreen 
                    ? '0 20px 50px rgba(0, 0, 0, 0.75), 0 0 35px rgba(56, 189, 248, 0.2)' 
                    : '0 10px 30px rgba(0, 0, 0, 0.45)',
                },
                canDragPiece: ({ piece }) => {
                  if (gameRef.current.isGameOver() || timeExpired || resigned) return false;
                  const turn = gameRef.current.turn();
                  const currentTurnPlayer = turn === 'w' ? whitePlayer : blackPlayer;
                  if (currentTurnPlayer !== 'manual') return false;
                  return piece.pieceType?.[0] === turn;
                },
                onPieceDrop: ({ sourceSquare, targetSquare }) => {
                  if (!targetSquare) return false;
                  return onDrop(sourceSquare, targetSquare);
                },
              }}
            />
          </div>

          <PlayerCard
            side={boardOrientation === 'white' ? 'white' : 'black'}
            playerType={bottomPlayer}
            isTurn={!topIsTurn}
            isThinking={isThinking}
            capturedPieces={bottomCaptured}
            responseMs={bottomResponseMs}
            timeRemaining={boardOrientation === 'white' ? whiteTime : blackTime}
            onResign={bottomPlayer === 'manual' ? () => setShowResignModal(true) : null}
          />

          {isFullscreen && (
            <div className="fs-floating-dock glass-panel">
              <button
                id="fs-play-pause-btn"
                className="fs-dock-btn"
                onClick={() => setIsPaused(p => !p)}
                title="Pause or resume match"
              >
                {isPaused ? '▶ Resume' : '⏸ Pause'}
              </button>
              {isPaused && (
                <button
                  id="fs-step-btn"
                  className="fs-dock-btn"
                  onClick={stepAIMove}
                  title="Step 1 AI move"
                >
                  ⏭ Step Move
                </button>
              )}
              <button
                id="fs-flip-btn"
                className="fs-dock-btn"
                onClick={() => setFlipBoard(f => !f)}
                title="Flip board orientation"
              >
                ⇅ Flip View
              </button>
              <button
                id="fs-sound-btn"
                className="fs-dock-btn"
                onClick={handleToggleSound}
                title="Toggle audio effects"
              >
                {soundMuted ? '🔇 Muted' : '🔊 Sound'}
              </button>
              <button
                id="fs-antigravity-btn"
                className={`fs-dock-btn ${isAntigravity ? 'ag-dock-active' : ''}`}
                onClick={toggleAntigravity}
                title="Toggle zero-gravity physics mode"
              >
                {isAntigravity ? '🌍 Gravity' : '🚀 Zero-G'}
              </button>
              {(whitePlayer === 'manual' || blackPlayer === 'manual') && (
                <button
                  id="fs-resign-btn"
                  className="fs-dock-btn fs-resign-btn"
                  onClick={() => setShowResignModal(true)}
                  title="Resign match"
                >
                  🏳️ Resign
                </button>
              )}
              <button
                id="fs-reset-btn"
                className="fs-dock-btn reset-btn-dock"
                onClick={onReset}
                title="Reset game"
              >
                ↺ Reset
              </button>
              <button
                id="fs-exit-btn"
                className="fs-dock-btn exit-btn-dock"
                onClick={toggleFullscreen}
                title="Exit fullscreen mode (Esc)"
              >
                ✕ Exit
              </button>
            </div>
          )}
        </section>

        {/* ── Right: Controls + Move History + Neural Panel ─────────────── */}
        {!isFullscreen && (
          <aside className="sidebar-section">
            <Controls
              whitePlayer={whitePlayer} setWhitePlayer={setWhitePlayer}
              blackPlayer={blackPlayer} setBlackPlayer={setBlackPlayer}
              difficulty={difficulty} setDifficulty={setDifficulty}
              timeControl={timeControl} setTimeControl={handleSetTimeControl}
              onResign={() => setShowResignModal(true)}
              isPaused={isPaused} setIsPaused={setIsPaused}
              stepAIMove={stepAIMove}
              moveDelay={moveDelay} setMoveDelay={setMoveDelay}
              onReset={onReset}
              onToggleFullscreen={toggleFullscreen}
              isAntigravity={isAntigravity}
              onToggleAntigravity={toggleAntigravity}
            />

            <div className="move-history glass-panel">
              <div className="move-history-header">
                <h3>Move History</h3>
                <span className="move-count">{moveHistory.length} moves</span>
              </div>

              <div className="moves-list" ref={movesContainerRef}>
                {moveHistory.length === 0 ? (
                  <div className="no-moves">No moves played yet. Start the match!</div>
                ) : (
                  moveHistory.reduce((acc, move, index) => {
                    if (index % 2 === 0) acc.push({ num: Math.floor(index / 2) + 1, white: move, black: '' });
                    else acc[acc.length - 1].black = move;
                    return acc;
                  }, []).map(t => (
                    <div key={t.num} className="move-row">
                      <span className="move-num">{t.num}.</span>
                      <span className="move-white">{t.white}</span>
                      <span className="move-black">{t.black}</span>
                    </div>
                  ))
                )}
              </div>

              {lastMove && (
                <div className="last-move-banner">
                  Last move: <strong>{lastMove.san}</strong> ({lastMove.from} → {lastMove.to})
                </div>
              )}
            </div>

            {neuralIsActive && (
              <NeuralNetPanel
                thinking={neuralThinking}
                isThinking={neuralScanning}
                lastChosenMove={neuralLastMove}
                learningStats={learningStats}
              />
            )}
          </aside>
        )}
      </main>

      {/* ── Pawn Promotion Selector Modal (FIDE Underpromotion Support) ── */}
      {pendingPromotion && (
        <div className="promotion-modal-overlay">
          <div className="promotion-modal glass-panel">
            <h3>Pawn Promotion</h3>
            <p>Select a piece to promote your pawn (FIDE Law 3.7e):</p>
            <div className="promotion-pieces-grid">
              {['q', 'r', 'b', 'n'].map(pieceKey => (
                <button
                  key={pieceKey}
                  id={`promo-piece-${pieceKey}`}
                  className="promotion-piece-card"
                  onClick={() => handleSelectPromotion(pieceKey)}
                >
                  <span className="promo-piece-glyph">
                    {PROMOTION_GLYPHS[pendingPromotion.color]?.[pieceKey] || '♕'}
                  </span>
                  <span className="promo-piece-name">
                    {PROMOTION_NAMES[pieceKey]}
                  </span>
                </button>
              ))}
            </div>
            <button 
              id="cancel-promotion-btn"
              className="cancel-promotion-btn" 
              onClick={() => setPendingPromotion(null)}
            >
              Cancel Move
            </button>
          </div>
        </div>
      )}

      {/* ── Resignation Confirmation Modal ── */}
      {showResignModal && (
        <div className="resign-modal-overlay">
          <div className="resign-modal glass-panel">
            <h3>Resign Match?</h3>
            <p>Are you sure you want to concede victory to your opponent?</p>
            <div className="modal-actions-row">
              <button 
                id="confirm-resign-btn"
                className="confirm-resign-btn" 
                onClick={confirmResign}
              >
                Yes, Resign Match
              </button>
              <button 
                id="cancel-resign-btn"
                className="cancel-resign-btn" 
                onClick={() => setShowResignModal(false)}
              >
                Keep Playing
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Semantic SEO Section (Googlebot & User Knowledge) ── */}
      {!isFullscreen && (
        <section className="seo-content-section glass-panel">
          <div className="seo-header">
            <h2>Free Online AI Chess Game – Play Neural Network &amp; Tactical Engines</h2>
            <span className="seo-badge">100% Free · No Sign-Up · Official FIDE Rules · Offline-Ready</span>
          </div>
          <p className="seo-lead">
            Play free chess online directly in your browser against intelligent AI opponents. Whether you want to test your tactical skills against our tournament-grade Minimax engine or experience an adaptive neural network that trains in real-time, AI Chess Arena delivers lightning-fast gameplay adhering strictly to official FIDE Laws of Chess with zero external cloud dependencies.
          </p>
          <div className="seo-grid">
            <article className="seo-card">
              <h3>🧠 In-Browser Deep Neural Network (ChessEvalNet)</h3>
              <p>
                Our neural network engine encodes the chess board across 773 tensor features (piece placements, color planes, turn perspective, castling rights) and evaluates positions using multi-layer dense activations. With client-side TD(0) temporal difference reinforcement learning, the engine adapts directly from your game decisions!
              </p>
            </article>
            <article className="seo-card">
              <h3>⚡ Tactical Minimax Engine with PeSTO Positional Tables</h3>
              <p>
                Engineered with Alpha-Beta pruning, dynamic capture/check move ordering, and PeSTO piece-square evaluation tables. Choose Tough mode for 3-ply tactical foresight or Easy mode for accessible casual play.
              </p>
            </article>
            <article className="seo-card">
              <h3>📜 Official FIDE Laws of Chess &amp; Special Moves</h3>
              <p>
                Complete tournament rule enforcement: Kingside (O-O) and Queenside (O-O-O) castling, en passant captures with immediate expiration, pawn promotion with underpromotion (Queen, Rook, Bishop, Knight), absolute pins, clock timeouts under FIDE Law 6.9, and auto-detection of draws (Stalemate, Threefold/Fivefold Repetition, 50/75-Move Rule, and Dead Positions).
              </p>
            </article>
            <article className="seo-card">
              <h3>🚀 Antigravity Zero-G Physics &amp; Fullscreen Arena</h3>
              <p>
                Break the laws of physics! Activate Antigravity mode to watch the chessboard and pieces float, drift, and tilt in zero gravity. Switch to the immersive Fullscreen Board View (<kbd>F</kbd> key) for tournament focus with customizable match clocks (Blitz, Rapid, Classical).
              </p>
            </article>
          </div>
        </section>
      )}
    </div>
  );
}
