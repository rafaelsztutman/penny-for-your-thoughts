"use client";

import type { CSSProperties } from "react";

export type PennyState = "idle" | "dropping" | "done";

type Props = {
  state: PennyState;
  onClick?: () => void;
  accent?: string;
};

/**
 * The penny coin.
 * - idle: inviting interactive coin that hovers
 * - dropping: spin-and-fall animation into a slot
 * - done: post-drop resting state (invisible)
 */
export function Penny({ state, onClick, accent = "#c97548" }: Props) {
  const dropping = state === "dropping" || state === "done";
  return (
    <div className={`penny-wrap penny-${state}`}>
      <div className="penny-shadow" />
      <button
        className="penny"
        onClick={onClick}
        disabled={dropping}
        aria-label="Spend a penny to see the thought"
        type="button"
      >
        <div className="penny-coin">
          <div
            className="penny-face penny-face-front"
            style={{ "--accent": accent } as CSSProperties}
          >
            <svg viewBox="0 0 100 100" width="100%" height="100%">
              <defs>
                <radialGradient id="copper-front" cx="35%" cy="30%" r="80%">
                  <stop offset="0%" stopColor="#f4c79c" />
                  <stop offset="40%" stopColor="#d99464" />
                  <stop offset="80%" stopColor="#a8643c" />
                  <stop offset="100%" stopColor="#6b3a1f" />
                </radialGradient>
                <radialGradient id="copper-rim" cx="50%" cy="50%" r="50%">
                  <stop offset="86%" stopColor="rgba(0,0,0,0)" />
                  <stop offset="93%" stopColor="rgba(0,0,0,0.35)" />
                  <stop offset="100%" stopColor="rgba(0,0,0,0.05)" />
                </radialGradient>
              </defs>
              <circle cx="50" cy="50" r="49" fill="url(#copper-front)" />
              <circle cx="50" cy="50" r="49" fill="url(#copper-rim)" />
              <circle
                cx="50"
                cy="50"
                r="44"
                fill="none"
                stroke="rgba(40,20,8,0.25)"
                strokeWidth="0.5"
              />
              <g transform="translate(50 52)" fill="rgba(40,20,8,0.45)">
                <path d="M-14 -18 Q-18 -10 -16 -2 Q-14 4 -12 8 L-12 16 L10 16 L10 8 Q14 6 16 0 Q18 -8 14 -16 Q8 -22 -2 -22 Q-10 -22 -14 -18 Z" />
                <circle cx="-3" cy="-8" r="1.5" fill="rgba(255,240,220,0.4)" />
              </g>
              <text
                x="50"
                y="92"
                textAnchor="middle"
                fontSize="6"
                fill="rgba(40,20,8,0.55)"
                fontFamily="var(--font-jetbrains-mono), monospace"
                letterSpacing="1.2"
              >
                ONE THOUGHT
              </text>
              <text
                x="50"
                y="14"
                textAnchor="middle"
                fontSize="5"
                fill="rgba(40,20,8,0.55)"
                fontFamily="var(--font-jetbrains-mono), monospace"
                letterSpacing="1.5"
              >
                · NLA ·
              </text>
            </svg>
          </div>
          <div className="penny-face penny-face-back">
            <svg viewBox="0 0 100 100" width="100%" height="100%">
              <circle cx="50" cy="50" r="49" fill="url(#copper-front)" />
              <circle cx="50" cy="50" r="49" fill="url(#copper-rim)" />
              <circle
                cx="50"
                cy="50"
                r="44"
                fill="none"
                stroke="rgba(40,20,8,0.25)"
                strokeWidth="0.5"
              />
              <g stroke="rgba(40,20,8,0.4)" fill="none" strokeWidth="0.6">
                <circle cx="50" cy="50" r="14" />
                <circle cx="50" cy="50" r="22" />
                <circle cx="50" cy="50" r="30" />
              </g>
              <circle cx="50" cy="50" r="3" fill="rgba(40,20,8,0.55)" />
              <text
                x="50"
                y="86"
                textAnchor="middle"
                fontSize="5"
                fill="rgba(40,20,8,0.55)"
                fontFamily="var(--font-jetbrains-mono), monospace"
                letterSpacing="1.2"
              >
                A · PENNY · FOR · YOUR
              </text>
            </svg>
          </div>
          <div className="penny-edge" />
        </div>
      </button>
    </div>
  );
}
