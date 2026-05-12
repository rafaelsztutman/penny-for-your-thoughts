import { generateObject } from "ai";
import { z } from "zod";
import type { NextRequest } from "next/server";
import { callModal, type ModalDecoding } from "@/lib/modal-client";
import { cacheGet, cacheKey, cacheSet } from "@/lib/cache";
import { clientIp, rateLimit } from "@/lib/ratelimit";
import { moderate } from "@/lib/moderation";
import { DEMO_ONLY, isGalleryPrompt } from "@/lib/gallery";

export const runtime = "nodejs";
export const maxDuration = 300;

const ThoughtSchema = z.object({
  answer: z.string().min(1),
  synthesis: z.string().min(1),
  tokens: z
    .array(
      z.object({
        token: z.string(),
        thoughts: z.array(z.string()).min(1).max(5),
      }),
    )
    .min(2)
    .max(40),
});

type ThoughtResult = z.infer<typeof ThoughtSchema>;

const RequestSchema = z.object({
  prompt: z.string().min(1).max(500),
  // Optional quality tier — 1 (default, current production), 2 (small bump),
  // 3 (high bump). Hidden, used to A/B the pre-cache strategy. Cache key
  // includes tier so tiers don't collide.
  tier: z.number().int().min(1).max(3).optional(),
});

const TIERS = {
  1: {
    targetChips: 16,
    maxTokens: 1024,
    avMaxNewTokens: 80,
    maxChips: 20,
    synthesisModel: "anthropic/claude-sonnet-4-6",
  },
  2: {
    targetChips: 24,
    maxTokens: 1024,
    avMaxNewTokens: 120,
    maxChips: 28,
    synthesisModel: "anthropic/claude-sonnet-4-6",
  },
  3: {
    targetChips: 32,
    maxTokens: 1024,
    avMaxNewTokens: 180,
    maxChips: 36,
    synthesisModel: "anthropic/claude-sonnet-4-6",
  },
} as const;

const DEFAULT_TIER = 3;

const SYNTHESIS_SYSTEM = `You are reshaping Natural Language Autoencoder (NLA) decodings from Anthropic's open-weight verbalizer into a viewer-friendly payload.

Inputs you receive:
- question: the user's prompt
- answer: the base model's full answer
- pre_response: a verbose NLA verbalization of the residual stream just before the answer began
- token_decodings: an ordered list of {token, position, surrounding_text, verbose} —
  "verbose" is the AV's full English explanation of that token's residual
  "surrounding_text" is the slice of the answer around that token (with the token itself marked in <<…>>) so you can see what the model was emitting at that exact point

Your job has two parts:

1. **synthesis** — a substantive second-person summary (180–280 words) of what the decodings actually surfaced about the model's thinking. Treat the rich AV verbalizations as your real source material — you are summarizing 30+ paragraphs of decoded thought into something a reader can absorb in 30 seconds.

   Structure (no headings; just 2–3 short paragraphs separated by blank lines):
   • **Opening (~50 words)**: orient the reader. What was the pre-response state? What was the model gearing up to do? Quote a striking phrase from the pre-response decoding if there is one.
   • **Middle (~100–150 words)**: surface the 3–5 most interesting concepts that emerged across the trajectory — name them concretely (e.g. "limbic-system imagery", "Kantian universalizability", "waggle-dance distance encoding", "Mufasa's death as grief shorthand"). Where the decoding shows a *shift* (from one framing to another, from generic to specific, from confident to hedging), call it out. Use single-quoted excerpts from the verbose blobs when a phrase is too good to paraphrase.
   • **Close (~30–60 words)**: what did the model land on? A short note on the synthesis or framing of the final tokens.

   Voice: present-tense, observational, second-person ("you can see…", "notice how at 'X' the activations…"). Avoid mysticism, never imply the model "felt" anything; this is what was decoded, not what was experienced. Avoid "first… then… by the time" chronological filler — prefer naming concepts directly.

2. **tokens** — for each token in token_decodings, produce 1–3 SHORT noun phrases (1–4 words each). These render as chips. Rules:
   - Use surrounding_text to GROUND the chips in what the model was actually saying at that point. Generic chips like "Q&A structure" are weakest; specific chips that reflect the *local* meaning ("chlorophyll", "wavelength absorption", "fall transition") are strongest.
   - VARY across tokens. If a concept already appeared in an earlier token's chips, do not repeat it — pick a different aspect of this token's verbose decoding instead.
   - PREFER substantive subject-matter concepts over stylistic/format concepts. Only use format chips ("Q&A tone", "structured explainer") when the verbose genuinely offers nothing else.
   - QUALITY over quantity. 1 specific chip is better than 3 generic ones.
   - Preserve token.text exactly as given.

Pass the answer through unchanged. Do not invent decodings — if a verbose blob is uninformative, produce best-effort tags rather than fabricating.`;

function surroundingText(
  answer: string,
  tokenText: string,
  occurrence: number,
  windowChars = 60,
): string {
  // Walk through `answer` to find the Nth occurrence of `tokenText`, then
  // return ±windowChars around it with the token wrapped in <<…>>. Falls back
  // to the whole answer truncated if the token can't be located cleanly
  // (e.g. when Gemma's tokenizer split a word across positions).
  if (!tokenText) return "";
  let pos = -1;
  let from = 0;
  for (let i = 0; i <= occurrence; i++) {
    pos = answer.indexOf(tokenText, from);
    if (pos < 0) break;
    from = pos + tokenText.length;
  }
  if (pos < 0) {
    return answer.length > windowChars * 2
      ? answer.slice(0, windowChars * 2) + "…"
      : answer;
  }
  const start = Math.max(0, pos - windowChars);
  const end = Math.min(answer.length, pos + tokenText.length + windowChars);
  const prefix = start > 0 ? "…" : "";
  const suffix = end < answer.length ? "…" : "";
  return (
    prefix +
    answer.slice(start, pos) +
    `<<${tokenText}>>` +
    answer.slice(pos + tokenText.length, end) +
    suffix
  );
}

function buildSynthesisPrompt(question: string, modal: ModalDecoding): string {
  const lines = [
    `User question: ${question}`,
    ``,
    `Base model answer:`,
    modal.response,
    ``,
    `Pre-response NLA decoding:`,
    modal.pre_response_thought ?? "(unavailable)",
    ``,
    `Per-token NLA decodings (in order):`,
  ];
  // Track occurrence count per token text so the Nth " the" gets the right
  // surrounding context, not the first " the" every time.
  const seen = new Map<string, number>();
  for (const t of modal.tokens) {
    const n = seen.get(t.text) ?? 0;
    seen.set(t.text, n + 1);
    const ctx = surroundingText(modal.response, t.text, n);
    lines.push(
      `- position=${t.idx ?? "?"} token=${JSON.stringify(t.text)}`,
      `    surrounding_text: ${JSON.stringify(ctx)}`,
      `    verbose: ${JSON.stringify(t.thought)}`,
    );
  }
  return lines.join("\n");
}

/**
 * Trim a base-model response to the last sentence boundary so we never display
 * a mid-sentence fragment when Gemma hits max_tokens. If the text already ends
 * cleanly (sentence punctuation, list marker, code fence), return it as-is.
 */
function trimToLastSentence(text: string): string {
  const trimmed = text.trimEnd();
  if (/[.!?…)"”』\]]\s*$/.test(trimmed)) return trimmed;
  if (/\*\*\s*$/.test(trimmed)) return trimmed;
  if (/```\s*$/.test(trimmed)) return trimmed;
  const win = trimmed.slice(-500);
  const re = /[.!?…](?=\s|$)/g;
  let last = -1;
  for (const m of win.matchAll(re)) {
    last = m.index ?? last;
  }
  if (last < 0) return trimmed;
  const cutoff = trimmed.length - win.length + last + 1;
  return trimmed.slice(0, cutoff);
}

async function buildResult(
  question: string,
  modal: ModalDecoding,
  cfg: (typeof TIERS)[keyof typeof TIERS],
): Promise<ThoughtResult> {
  const cleanAnswer = trimToLastSentence(modal.response);
  const cleanModal: ModalDecoding = {
    ...modal,
    response: cleanAnswer,
    tokens:
      modal.tokens.length > cfg.maxChips
        ? modal.tokens.slice(0, cfg.maxChips)
        : modal.tokens,
  };
  const { object } = await generateObject({
    model: cfg.synthesisModel,
    schema: ThoughtSchema,
    system: SYNTHESIS_SYSTEM,
    prompt: buildSynthesisPrompt(question, cleanModal),
  });
  return { ...object, answer: cleanAnswer };
}

export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "invalid json" }, { status: 400 });
  }
  const parsed = RequestSchema.safeParse(body);
  if (!parsed.success) {
    return Response.json({ error: "invalid request" }, { status: 400 });
  }
  const prompt = parsed.data.prompt;

  // Defense-in-depth: when the demo is locked to gallery-only, reject
  // anything that doesn't match a curated prompt before any cache /
  // rate-limit / moderation / Modal cost is incurred. The UI hides the
  // open input field in this mode, but a hand-crafted POST would still
  // reach this route.
  if (DEMO_ONLY && !isGalleryPrompt(prompt)) {
    return Response.json(
      {
        error: "rejected",
        message:
          "This demo is locked to its gallery questions. Pick one from the list to see how it works.",
      },
      { status: 400 },
    );
  }

  const tier = parsed.data.tier ?? DEFAULT_TIER;
  const cfg = TIERS[tier as keyof typeof TIERS];
  // Tier in key so experiments don't poison the default cache.
  const key =
    tier === DEFAULT_TIER ? cacheKey(prompt) : `${cacheKey(prompt)}:t${tier}`;

  // Cache lookup happens before the rate-limit charge — a repeated question
  // from the same IP doesn't burn their budget since it costs us nothing.
  const cached = await cacheGet<ThoughtResult>(key);
  if (cached) {
    return Response.json(cached, { headers: { "x-cache": "hit" } });
  }

  const limit = await rateLimit(clientIp(req));
  if (!limit.allowed) {
    return Response.json(
      { error: "rate_limited", window: limit.window, retry_after: limit.resetSeconds },
      {
        status: 429,
        headers: {
          "retry-after": String(limit.resetSeconds ?? 60),
          "x-ratelimit-window": String(limit.window ?? ""),
        },
      },
    );
  }

  // Moderate after rate-limit so abusers still consume their budget. Cheap
  // Haiku call; fails open on transient errors (see lib/moderation.ts).
  const verdict = await moderate(prompt);
  if (!verdict.safe) {
    return Response.json(
      {
        error: "rejected",
        message:
          verdict.reason ??
          "Let's try a different question — this one falls outside what the demo will answer.",
      },
      { status: 400 },
    );
  }

  let modal: ModalDecoding;
  try {
    modal = await callModal(prompt, {
      targetChips: cfg.targetChips,
      maxTokens: cfg.maxTokens,
      avMaxNewTokens: cfg.avMaxNewTokens,
    });
  } catch (err) {
    console.error("modal call failed:", err);
    return Response.json({ error: "decode failed" }, { status: 502 });
  }

  let result: ThoughtResult;
  try {
    result = await buildResult(prompt, modal, cfg);
  } catch (err) {
    console.error("synthesis failed:", err);
    return Response.json({ error: "synthesis failed" }, { status: 502 });
  }

  await cacheSet(key, result);
  return Response.json(result, {
    headers: { "x-cache": "miss", "x-tier": String(tier) },
  });
}
