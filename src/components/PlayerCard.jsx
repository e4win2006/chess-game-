import React from 'react';

const PIECE_SYMBOLS = {
  p: '♟',
  n: '♞',
  b: '♝',
  r: '♜',
  q: '♛',
  k: '♚',
};

function getPlayerMeta(playerType) {
  switch (playerType) {
    case 'manual':    return { name: 'Human Player',           badge: 'HUMAN',     cls: 'badge-human',   avatar: null };
    case 'neural':    return { name: 'Neural Net (Evolved)',   badge: 'NET',       cls: 'badge-neural',  avatar: '🧠' };
    case 'minimax':   return { name: 'Tactical Minimax AI',    badge: 'MINIMAX',   cls: 'badge-ai',      avatar: '⚡' };
    case 'heuristic': return { name: 'Positional Heuristic AI',badge: 'HEURISTIC', cls: 'badge-ai',      avatar: '🎯' };
    default:          return { name: playerType,               badge: 'AI',        cls: 'badge-ai',      avatar: '🤖' };
  }
}

function formatClockTime(seconds) {
  if (seconds === null || seconds === undefined) return null;
  const m = Math.floor(Math.max(0, seconds) / 60);
  const s = Math.max(0, seconds) % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export default function PlayerCard({
  side,
  playerType,
  isTurn,
  isThinking,
  capturedPieces = [],
  responseMs = null,
  timeRemaining = null,
  onResign = null,
}) {
  const isWhite = side === 'white';
  const { name, badge, cls, avatar } = getPlayerMeta(playerType);
  const formattedTime = formatClockTime(timeRemaining);
  const isLowTime = timeRemaining !== null && timeRemaining <= 30;

  return (
    <div className={`player-card glass-panel ${isTurn ? 'active-turn' : ''} ${playerType === 'neural' && isTurn ? 'neural-turn' : ''}`}>
      <div className="player-info">
        <div className={`player-avatar ${isWhite ? 'avatar-white' : 'avatar-black'} ${playerType === 'neural' ? 'avatar-neural' : ''}`}>
          {avatar || (isWhite ? '♔' : '♚')}
        </div>
        <div className="player-details">
          <div className="player-name-row">
            <span className="player-name">{name}</span>
            <span className={`player-badge ${cls}`}>{badge}</span>
            {onResign && (
              <button 
                className="card-resign-btn" 
                onClick={onResign}
                title={`Resign game as ${side}`}
              >
                🏳️ Resign
              </button>
            )}
          </div>
          <div className="player-side-label">
            {isWhite ? 'White Pieces (Plays 1st)' : 'Black Pieces (Plays 2nd)'}
          </div>
        </div>
      </div>

      <div className="player-meta">
        <div className="player-meta-row">
          {formattedTime !== null && (
            <div className={`player-clock-badge ${isLowTime ? 'clock-danger' : ''} ${isTurn ? 'clock-active' : ''}`}>
              <span className="clock-icon">⏱</span>
              <span className="clock-digits">{formattedTime}</span>
            </div>
          )}
          {isThinking && isTurn && (
            <div className={`thinking-indicator ${playerType === 'neural' ? 'thinking-neural' : ''}`}>
              <span className="pulse-dot" />
              {playerType === 'neural' ? 'Neural scanning…' : 'Thinking…'}
            </div>
          )}
          {responseMs !== null && (
            <span className="response-time-badge" title="Last response time">
              ⚡ {responseMs < 1000 ? `${responseMs}ms` : `${(responseMs/1000).toFixed(1)}s`}
            </span>
          )}
        </div>

        <div className="captured-pieces">
          {capturedPieces.map((piece, idx) => (
            <span key={idx} className="captured-symbol">
              {PIECE_SYMBOLS[piece.toLowerCase()] || piece}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
