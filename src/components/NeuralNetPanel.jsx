import React, { useState } from 'react';
import { NET_INFO } from '../services/neuralNetwork';

/**
 * Color-codes a score in [-1,1]:
 *   ≥ +0.3  → emerald (good for White)
 *   ≤ −0.3  → rose    (good for Black)
 *   else    → yellow  (roughly even)
 */
function scoreColor(score) {
  if (score >= 0.3)  return '#10b981';
  if (score <= -0.3) return '#f43f5e';
  return '#eab308';
}

function ScoreBar({ score }) {
  // Map [-1,1] → [0%,100%] fill width, centred at 50%
  const pct      = ((score + 1) / 2) * 100;
  const barColor = scoreColor(score);
  return (
    <div className="nn-score-bar-track">
      <div
        className="nn-score-bar-fill"
        style={{ width: `${Math.round(pct)}%`, background: barColor }}
      />
    </div>
  );
}

export default function NeuralNetPanel({ thinking, isThinking, lastChosenMove }) {
  const [open, setOpen] = useState(true);

  const hasData = thinking && thinking.length > 0;

  return (
    <div className={`nn-panel glass-panel ${open ? 'nn-panel-open' : 'nn-panel-closed'}`}>
      {/* ── Header / Toggle ─────────────────────────────────────── */}
      <button
        className="nn-panel-toggle"
        onClick={() => setOpen(o => !o)}
        aria-expanded={open}
        title={open ? 'Collapse Neural Net panel' : 'Expand Neural Net panel'}
      >
        <span className="nn-panel-icon">🧠</span>
        <span className="nn-panel-title">Neural Net Thinking</span>
        {isThinking && (
          <span className="nn-scanning-badge">
            <span className="pulse-dot" />
            Scanning…
          </span>
        )}
        {!isThinking && lastChosenMove && !open && (
          <span className="nn-chosen-inline">▶ {lastChosenMove}</span>
        )}
        <span className="nn-chevron">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="nn-panel-body">
          {/* Architecture info */}
          <div className="nn-arch-info">{NET_INFO}</div>

          {/* Thinking / empty states */}
          {isThinking && (
            <div className="nn-thinking-state">
              <div className="nn-thinking-dots">
                <span /><span /><span />
              </div>
              <span>Evaluating all legal moves…</span>
            </div>
          )}

          {!isThinking && !hasData && (
            <div className="nn-empty">
              No evaluations yet — Neural Net hasn't moved.
            </div>
          )}

          {/* Move evaluation table */}
          {!isThinking && hasData && (
            <>
              <div className="nn-chosen-banner">
                Chosen move: <strong>{lastChosenMove}</strong>
                <span className="nn-best-tag">BEST</span>
              </div>

              <div className="nn-table-header">
                <span>Move</span>
                <span>Score</span>
                <span>Evaluation</span>
              </div>

              <div className="nn-rows-scroll">
                {thinking.map((entry, i) => (
                  <div
                    key={`${entry.san}-${i}`}
                    className={`nn-row ${entry.isBest ? 'nn-row-best' : ''}`}
                  >
                    {/* Rank + SAN */}
                    <div className="nn-row-move">
                      <span className="nn-rank">#{i + 1}</span>
                      <span
                        className="nn-san"
                        style={{ color: entry.isBest ? '#10b981' : undefined }}
                      >
                        {entry.san}
                      </span>
                      {entry.isCheckmate && (
                        <span className="nn-tag-mate">MATE</span>
                      )}
                      {entry.isBest && (
                        <span className="nn-tag-best">✓</span>
                      )}
                    </div>

                    {/* Numeric score */}
                    <div
                      className="nn-row-score"
                      style={{ color: scoreColor(entry.score) }}
                    >
                      {entry.score >= 0 ? '+' : ''}{entry.score.toFixed(3)}
                    </div>

                    {/* Score bar */}
                    <div className="nn-row-bar">
                      <ScoreBar score={entry.score} />
                    </div>
                  </div>
                ))}
              </div>

              <div className="nn-footer">
                {thinking.length} moves evaluated · 1-ply lookahead
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
