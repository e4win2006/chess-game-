import React, { useState } from 'react';

export default function Controls({
  whitePlayer, setWhitePlayer,
  blackPlayer, setBlackPlayer,
  difficulty, setDifficulty,
  timeControl, setTimeControl,
  onResign,
  isPaused, setIsPaused,
  stepAIMove,
  moveDelay, setMoveDelay,
  onReset,
  onToggleFullscreen,
  isAntigravity,
  onToggleAntigravity,
}) {
  const [showRules, setShowRules] = useState(false);
  const isAIVsAI = whitePlayer !== 'manual' && blackPlayer !== 'manual';
  const hasHumanPlayer = whitePlayer === 'manual' || blackPlayer === 'manual';

  return (
    <div className="controls glass-panel">
      <div className="panel-header">
        <h2>Match Settings</h2>
        <button 
          id="rules-toggle-btn"
          className="rules-toggle-btn"
          onClick={() => setShowRules(!showRules)}
          title="Toggle Complete Official FIDE Chess Rules Guide"
        >
          {showRules ? 'Hide Rules ✕' : 'FIDE Rules 📖'}
        </button>
      </div>

      {showRules && (
        <div className="rules-card">
          <div className="rules-header">
            <h4>Official FIDE Laws of Chess Reference</h4>
            <span className="rules-tag">Tournament Standard</span>
          </div>
          
          <div className="rules-sections-container">
            <div className="rule-box">
              <h5>1. Basic Movement &amp; Captures</h5>
              <ul>
                <li><strong>White moves first:</strong> Players strictly alternate turns.</li>
                <li><strong>Rook:</strong> Moves horizontally or vertically any distance. No jumping.</li>
                <li><strong>Bishop:</strong> Moves diagonally any distance along its square color. No jumping.</li>
                <li><strong>Queen:</strong> Combines straight and diagonal movements. No jumping.</li>
                <li><strong>Knight:</strong> Moves in an "L" shape (2 squares then 1 perpendicular). The <em>only</em> piece that jumps over intervening pieces.</li>
                <li><strong>Pawn:</strong> Moves 1 square forward (or 2 squares on its very first move). Captures 1 square diagonally forward.</li>
                <li><strong>King:</strong> Moves 1 square in any direction to safe, non-attacked squares.</li>
              </ul>
            </div>

            <div className="rule-box">
              <h5>2. King Safety, Check &amp; Absolute Pins</h5>
              <ul>
                <li><strong>Illegal Checks:</strong> You can never make a move that leaves or places your King in check.</li>
                <li><strong>Absolute Pins:</strong> A piece shielding its King from an enemy line of fire cannot move off the pin line.</li>
                <li><strong>Attacking Squares:</strong> Pinned enemy pieces still attack squares and prevent your King from stepping into them.</li>
              </ul>
            </div>

            <div className="rule-box">
              <h5>3. Castling (Kingside O-O &amp; Queenside O-O-O)</h5>
              <ul>
                <li>King moves 2 squares toward Rook; Rook jumps over King to adjacent square.</li>
                <li>Legal ONLY if neither King nor that Rook has moved previously in the game.</li>
                <li>All squares between King and Rook must be empty.</li>
                <li>King cannot be in check, cross through check, or land in check. (Rook passing through an attacked square on b1/b8 is legal).</li>
              </ul>
            </div>

            <div className="rule-box">
              <h5>4. En Passant &amp; Pawn Promotion</h5>
              <ul>
                <li><strong>En Passant:</strong> If an enemy pawn advances 2 squares from start and lands adjacent, you can capture it diagonally as if it moved 1 square. Expires immediately on the next turn.</li>
                <li><strong>Pawn Promotion &amp; Underpromotion:</strong> Reaching 8th rank (White) or 1st rank (Black) promotes immediately to Queen, Rook, Bishop, or Knight. Underpromotion is fully supported!</li>
              </ul>
            </div>

            <div className="rule-box">
              <h5>5. Game Termination &amp; Clock Timeout</h5>
              <ul>
                <li><strong>Checkmate:</strong> King is in check with zero legal escape moves.</li>
                <li><strong>Resignation:</strong> A player may concede victory to the opponent at any point.</li>
                <li><strong>Clock Timeout (FIDE Law 6.9):</strong> When a player runs out of time, opponent wins—<em>unless</em> opponent lacks sufficient mating material, in which case the game is drawn.</li>
              </ul>
            </div>

            <div className="rule-box">
              <h5>6. Mandatory Draw Conditions</h5>
              <ul>
                <li><strong>Stalemate:</strong> Player to move is not in check, but has zero legal moves.</li>
                <li><strong>Threefold &amp; Fivefold Repetition:</strong> Same position occurs 3 times (draw) or 5 times (FIDE mandatory draw).</li>
                <li><strong>50-Move &amp; 75-Move Rule:</strong> 50 moves without capture/pawn advance (draw claim) or 75 moves (mandatory draw).</li>
                <li><strong>Insufficient Material (Dead Position):</strong> K vs K, K+B vs K, K+N vs K, and K+B vs K+B with same-colored bishops.</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      <div className="setting-group">
        <label htmlFor="white-player-select">White Player (First Turn):</label>
        <select 
          id="white-player-select"
          value={whitePlayer} 
          onChange={(e) => setWhitePlayer(e.target.value)}
        >
          <option value="manual">👤 Human (Manual)</option>
          <option value="neural">🧠 Neural Net (Evolved Local AI)</option>
          <option value="minimax">⚡ Tactical Minimax (Offline AI)</option>
          <option value="heuristic">🎯 Positional Heuristic (Offline AI)</option>
        </select>
      </div>

      <div className="setting-group">
        <label htmlFor="black-player-select">Black Player:</label>
        <select 
          id="black-player-select"
          value={blackPlayer} 
          onChange={(e) => setBlackPlayer(e.target.value)}
        >
          <option value="manual">👤 Human (Manual)</option>
          <option value="neural">🧠 Neural Net (Evolved Local AI)</option>
          <option value="minimax">⚡ Tactical Minimax (Offline AI)</option>
          <option value="heuristic">🎯 Positional Heuristic (Offline AI)</option>
        </select>
      </div>

      <div className="setting-group">
        <label htmlFor="time-control-select">⏳ Match Clock / Time Control:</label>
        <select
          id="time-control-select"
          value={timeControl}
          onChange={(e) => setTimeControl(e.target.value)}
        >
          <option value="none">Casual / Unlimited (No Clock)</option>
          <option value="180">⚡ 3 Minutes (Blitz)</option>
          <option value="300">⏱ 5 Minutes (Rapid)</option>
          <option value="600">🏛 10 Minutes (Classical)</option>
        </select>
      </div>

      <div className="setting-group">
        <label>AI Difficulty:</label>
        <div className="toggle-group">
          <button 
            id="difficulty-easy-btn"
            className={`toggle-btn ${difficulty === 'easy' ? 'active' : ''}`} 
            onClick={() => setDifficulty('easy')}
          >
            Easy Mode
          </button>
          <button 
            id="difficulty-tough-btn"
            className={`toggle-btn ${difficulty === 'tough' ? 'active' : ''}`} 
            onClick={() => setDifficulty('tough')}
          >
            Tough Mode
          </button>
        </div>
      </div>

      {isAIVsAI && (
        <div className="spectator-section">
          <label>Spectator Pace / Delay:</label>
          <div className="speed-buttons">
            <button 
              className={`speed-btn ${moveDelay === 1000 ? 'active' : ''}`}
              onClick={() => setMoveDelay(1000)}
            >
              1.0s (Fast)
            </button>
            <button 
              className={`speed-btn ${moveDelay === 2000 ? 'active' : ''}`}
              onClick={() => setMoveDelay(2000)}
            >
              2.0s (Normal)
            </button>
            <button 
              className={`speed-btn ${moveDelay === 3500 ? 'active' : ''}`}
              onClick={() => setMoveDelay(3500)}
            >
              3.5s (Slow)
            </button>
          </div>

          <div className="spectator-actions">
            <button 
              id="ai-play-pause-btn"
              className={`action-btn ${isPaused ? 'play-mode' : 'pause-mode'}`}
              onClick={() => setIsPaused(!isPaused)}
            >
              {isPaused ? '▶ Start / Resume AI Match' : '⏸ Pause AI Match'}
            </button>
            {isPaused && (
              <button 
                id="ai-step-btn"
                className="action-btn step-mode"
                onClick={stepAIMove}
              >
                ⏭ Step 1 Move
              </button>
            )}
          </div>
        </div>
      )}

      <div className="controls-action-row">
        {hasHumanPlayer && onResign && (
          <button 
            id="resign-game-btn"
            className="resign-btn" 
            onClick={onResign}
            title="Concede match by resignation"
          >
            🏳️ Resign Match
          </button>
        )}

        <button 
          id="antigravity-toggle-btn"
          className={`antigravity-mode-btn ${isAntigravity ? 'ag-active' : ''}`}
          onClick={onToggleAntigravity}
          title="Toggle Zero-Gravity physics mode (or type 'antigravity')"
        >
          {isAntigravity ? '🌍 Restore Normal Gravity' : '🚀 Antigravity Mode (Zero-G)'}
        </button>

        <button 
          id="fullscreen-toggle-btn" 
          className="fullscreen-mode-btn" 
          onClick={onToggleFullscreen}
          title="Switch to immersive Fullscreen Board view (or press F)"
        >
          ⛶ Fullscreen Board Mode
        </button>

        <button id="reset-game-btn" className="reset-btn" onClick={onReset}>
          ↺ New Game / Reset Board
        </button>
      </div>
    </div>
  );
}
