"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { NeuralNetwork } from "@/components/NeuralNetwork";
import { Penny } from "@/components/Penny";
import type { ThoughtResult } from "@/lib/types";
import { DEMO_ONLY, GALLERY_PROMPTS } from "@/lib/gallery";

const ACCENT = "#c97548";
const SHOW_RAW_BY_DEFAULT = true;

const EXAMPLE_PROMPTS = GALLERY_PROMPTS;

type Phase =
  | "idle"
  | "loading"
  | "answered"
  | "spending"
  | "revealing"
  | "revealed"
  | "rejected";

type FetchOutcome =
  | { kind: "ok"; data: ThoughtResult }
  | { kind: "rejected"; message: string }
  | { kind: "error" };

function mockResponse(): ThoughtResult {
  return {
    answer:
      "I'm thinking about your question — but the bridge to the model's interior monologue is offline. Try again in a moment, or check the README for setup.",
    synthesis:
      "The model is in a placeholder state. Its activations would normally form a coherent narrative here — first orienting to the question, then retrieving relevant concepts, then composing a structured answer.",
    tokens: [
      { token: "I'm", thoughts: ["self-reference", "first-person assistant frame"] },
      { token: " thinking", thoughts: ["cognition", "internal process"] },
      { token: " about", thoughts: ["relational preposition", "focus shift"] },
      { token: " your", thoughts: ["address user", "ownership"] },
      { token: " question", thoughts: ["query frame", "interrogative"] },
    ],
  };
}

async function fetchThought(prompt: string): Promise<FetchOutcome> {
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (res.status === 400) {
      const body = (await res.json().catch(() => null)) as
        | { error?: string; message?: string }
        | null;
      if (body?.error === "rejected") {
        return {
          kind: "rejected",
          message:
            body.message ??
            "Let's try a different question — this one falls outside what the demo will answer.",
        };
      }
    }
    if (res.status === 429) {
      return {
        kind: "rejected",
        message:
          "You're sending questions faster than the demo can keep up — give it a moment and try again.",
      };
    }
    if (!res.ok) return { kind: "error" };
    const data = (await res.json()) as ThoughtResult;
    if (!data.answer || !Array.isArray(data.tokens)) return { kind: "error" };
    return { kind: "ok", data };
  } catch (err) {
    console.warn("ask failed:", err);
    return { kind: "error" };
  }
}

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [data, setData] = useState<ThoughtResult | null>(null);
  const [rejectionMsg, setRejectionMsg] = useState("");
  const [rawOpen, setRawOpen] = useState(false);
  const [answerTyped, setAnswerTyped] = useState("");
  const [synthTyped, setSynthTyped] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  const activity =
    phase === "revealing" || phase === "revealed" || phase === "spending"
      ? 1
      : phase === "loading"
        ? 0.3
        : 0;

  // Fade the fixed network-stage as the user scrolls so it doesn't bleed
  // through the answer/synthesis text. Fully opaque at top, gone by half a
  // viewport down. CSS reads --network-opacity.
  useEffect(() => {
    const update = () => {
      const op = Math.max(
        0,
        1 - window.scrollY / (window.innerHeight * 0.5),
      );
      document.documentElement.style.setProperty(
        "--network-opacity",
        String(op),
      );
    };
    window.addEventListener("scroll", update, { passive: true });
    update();
    return () => window.removeEventListener("scroll", update);
  }, []);

  // Typewriter for answer
  useEffect(() => {
    if (phase !== "answered" || !data) return;
    setAnswerTyped("");
    let i = 0;
    const text = data.answer;
    const id = setInterval(() => {
      i += Math.max(1, Math.round(text.length / 80));
      setAnswerTyped(text.slice(0, i));
      if (i >= text.length) clearInterval(id);
    }, 28);
    return () => clearInterval(id);
  }, [phase, data]);

  // Typewriter for synthesis
  useEffect(() => {
    if (phase !== "revealing" || !data) return;
    setSynthTyped("");
    let i = 0;
    const text = data.synthesis;
    const id = setInterval(() => {
      i += Math.max(1, Math.round(text.length / 120));
      setSynthTyped(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(id);
        setPhase("revealed");
        if (SHOW_RAW_BY_DEFAULT) setRawOpen(true);
      }
    }, 22);
    return () => clearInterval(id);
  }, [phase, data]);

  const submit = useCallback(
    async (q?: string) => {
      const text = (q ?? prompt).trim();
      if (!text || phase === "loading") return;
      setPrompt(text);
      setPhase("loading");
      setData(null);
      setRejectionMsg("");
      setRawOpen(false);
      setSynthTyped("");
      setAnswerTyped("");
      const outcome = await fetchThought(text);
      if (outcome.kind === "rejected") {
        setRejectionMsg(outcome.message);
        setPhase("rejected");
        return;
      }
      const result = outcome.kind === "ok" ? outcome.data : mockResponse();
      setData(result);
      setPhase("answered");
    },
    [prompt, phase],
  );

  const spendPenny = () => {
    setPhase("spending");
    setTimeout(() => setPhase("revealing"), 2400);
  };

  const reset = () => {
    setPhase("idle");
    setData(null);
    setPrompt("");
    setRejectionMsg("");
    setAnswerTyped("");
    setSynthTyped("");
    setRawOpen(false);
    setTimeout(() => inputRef.current?.focus(), 50);
  };

  return (
    <div className="app" style={{ ["--accent" as string]: ACCENT }}>
      {/* Persistent network viz */}
      <div className="network-stage">
        <NeuralNetwork activity={activity} accent={ACCENT} />
        <div className="network-mask" />
        <div className="network-label">
          <span className="dot" />
          <span>
            residual stream · 7 layers ·{" "}
            {phase === "idle"
              ? "idle"
              : phase === "loading"
                ? "inferring"
                : phase === "revealing" || phase === "revealed"
                  ? "decoded"
                  : "ready"}
          </span>
        </div>
      </div>

      {/* Header */}
      <header className="hdr">
        <div className="hdr-mark">
          <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
            <circle
              cx="12"
              cy="12"
              r="10"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.2"
            />
            <circle
              cx="12"
              cy="12"
              r="5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.2"
            />
            <circle cx="12" cy="12" r="1.2" fill="currentColor" />
          </svg>
          <span>A Penny for Your Thoughts</span>
        </div>
      </header>

      {/* Main column */}
      <main className="col">
        {phase === "idle" && (
          <div className="intro">
            <h1>
              Ask a small question.
              <br />
              Then peek at what the model was thinking.
            </h1>
            <p className="intro-sub">
              An AI model answers your question. For a virtual penny, you can
              watch its thoughts translated back into language by a verbalizer.
              First as a narrative, then as the raw concepts at each token.
            </p>
          </div>
        )}

        {prompt && phase !== "idle" && (
          <div className="qa-question">
            <span className="qa-label">you asked</span>
            <p>{prompt}</p>
          </div>
        )}

        {phase === "loading" && (
          <div className="loading">
            <span className="loading-dots">
              <i />
              <i />
              <i />
            </span>
            <span>the agent is thinking</span>
          </div>
        )}

        {phase === "rejected" && (
          <div className="rejected">
            <span className="qa-label">demo declined</span>
            <p className="rejected-text">{rejectionMsg}</p>
            <div className="reset-row">
              <button className="reset-btn" onClick={reset} type="button">
                ask another →
              </button>
            </div>
          </div>
        )}

        {(phase === "answered" ||
          phase === "spending" ||
          phase === "revealing" ||
          phase === "revealed") &&
          data && (
            <div className="qa-answer">
              <span className="qa-label">the agent replied</span>
              <div className="answer-text markdown">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {answerTyped}
                </ReactMarkdown>
              </div>
            </div>
          )}

        {phase === "answered" && data && (
          <div className="penny-prompt">
            <p className="penny-label">A penny for its thoughts?</p>
            <Penny state="idle" onClick={spendPenny} accent={ACCENT} />
            <p className="penny-hint">click to spend</p>
          </div>
        )}

        {(phase === "spending" || phase === "revealing" || phase === "revealed") && (
          <div className="penny-drop-zone">
            <Penny
              state={phase === "spending" ? "dropping" : "done"}
              accent={ACCENT}
            />
            {(phase === "revealing" || phase === "revealed") && (
              <div className="slot-line">
                <span className="slot-mark">↓</span>
                <span>thought unlocked</span>
              </div>
            )}
          </div>
        )}

        {(phase === "revealing" || phase === "revealed") && data && (
          <div className="synthesis">
            <span className="qa-label accent">what it was thinking</span>
            <div className="synth-text markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {synthTyped}
              </ReactMarkdown>
            </div>

            {phase === "revealed" && (
              <div className="raw-panel">
                <button
                  className="raw-toggle"
                  onClick={() => setRawOpen(!rawOpen)}
                  aria-expanded={rawOpen}
                  type="button"
                >
                  <span className="chev" data-open={rawOpen}>
                    ▸
                  </span>
                  <span>raw thoughts · per token</span>
                  <span className="raw-count">{data.tokens.length} tokens</span>
                </button>
                {rawOpen && (
                  <div className="raw-body">
                    <div className="raw-hint">
                      Each token of the answer, paired with the concepts decoded
                      from the residual stream at that position. This is what the
                      NLA actually produces; the synthesis above is Claude&apos;s
                      reading of it.
                    </div>
                    <ol className="token-list">
                      {data.tokens.map((tk, i) => (
                        <li key={i} className="token-row">
                          <span className="token-idx">
                            {String(i).padStart(2, "0")}
                          </span>
                          <span className="token-text">
                            {tk.token.replace(/ /g, "·")}
                          </span>
                          <span className="token-thoughts">
                            {tk.thoughts.map((th, j) => (
                              <span key={j} className="thought-chip">
                                {th}
                              </span>
                            ))}
                          </span>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            )}

            {phase === "revealed" && (
              <div className="reset-row">
                <button className="reset-btn" onClick={reset} type="button">
                  ask another →
                </button>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Dock — only when idle. In demo mode the open input is hidden and
          the curated gallery is the only way in. */}
      {phase === "idle" && (
        <div className="dock">
          {!DEMO_ONLY && (
            <form
              className="input-row"
              onSubmit={(e) => {
                e.preventDefault();
                submit();
              }}
            >
              <input
                ref={inputRef}
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="ask the agent something…"
                autoFocus
              />
              <button type="submit" disabled={!prompt.trim()}>
                ask
                <svg width="14" height="14" viewBox="0 0 24 24">
                  <path
                    d="M5 12h14M13 5l7 7-7 7"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
            </form>
          )}
          <div className="examples">
            <span className="examples-label">
              {DEMO_ONLY ? "pick one" : "or try"}
            </span>
            {EXAMPLE_PROMPTS.map((p, i) => (
              <button
                key={i}
                className="example-chip"
                onClick={() => submit(p)}
                type="button"
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      )}

      <footer className="ftr">
        <div className="ftr-row">
          <span className="ftr-disclaimer">
            Built for learning · not a production AI service · not affiliated
            with Anthropic
          </span>
        </div>
        <div className="ftr-row">
          <span>
            Based on Anthropic&apos;s{" "}
            <a
              href="https://transformer-circuits.pub/2026/nla/index.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              <em>Natural Language Autoencoders</em>
            </a>
            {" — "}
            <a
              href="https://transformer-circuits.pub/2026/nla/index.html"
              target="_blank"
              rel="noopener noreferrer"
            >
              paper ↗
            </a>{" · "}
            <a
              href="https://www.anthropic.com/research/natural-language-autoencoders"
              target="_blank"
              rel="noopener noreferrer"
            >
              blog ↗
            </a>{" · "}
            <a
              href="https://www.youtube.com/watch?v=j2knrqAzYVY"
              target="_blank"
              rel="noopener noreferrer"
            >
              video ↗
            </a>
          </span>
        </div>
        <div className="ftr-row">
          <span>
            Gemma-3-12B-IT · synthesis by Claude Sonnet 4.6 · NLA recipe from{" "}
            <a
              href="https://github.com/kitft/natural_language_autoencoders"
              target="_blank"
              rel="noopener noreferrer"
            >
              kitft/natural_language_autoencoders ↗
            </a>
          </span>
          <span className="ftr-sep">·</span>
          <a
            href="https://rafaeloliveira.xyz"
            target="_blank"
            rel="noopener noreferrer"
          >
            rafaeloliveira.xyz ↗
          </a>
        </div>
      </footer>
    </div>
  );
}
