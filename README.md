# A Penny for Your Thoughts

A visual demo of Anthropic's [Natural Language Autoencoders](https://transformer-circuits.pub/2026/nla/index.html). Ask the agent a question, then spend a virtual penny to peek at what it was thinking — first as a synthesized narrative, then as the raw per-token activation decodings.

**Live:** https://penny.rafaeloliveira.xyz

## How it works

The agent is **Gemma-3-12B-IT** (Google's open-weight 12B instruct model). When you ask a question, this happens server-side:

1. The browser POSTs the question to `/api/ask`.
2. The route checks an Upstash Redis cache (key: `sha256(normalized question)`). Repeat questions return in ~1s.
3. On miss: the route runs an IP-based rate limit, screens the prompt with a Claude Haiku 4.5 moderation pass, then calls a Modal H100 service that:
   - Generates Gemma's answer (up to 1024 tokens, sentence-trimmed if it hits the cap).
   - Captures Gemma's residual-stream activation at layer 32 at the last input token plus every response token.
   - Downsamples to ~32 evenly-spaced positions across the actual answer (short answers get dense sampling; long answers, sparse).
   - Verbalizes each captured activation through `kitft/nla-gemma3-12b-L32-av` — Anthropic's released NLA verbalizer — into a natural-language description.
4. Claude Sonnet 4.6 (via the Vercel AI Gateway) reshapes the verbalizer's verbose English blobs into a 200–280 word grounded narrative plus 1–3 short concept chips per token.
5. The result is cached for ~1 year and returned.

Modal stays scaled to zero between requests; cold starts after a populated HF volume cache are ~30–40s.

## Stack

- **Frontend:** Next.js 16 (App Router, Turbopack), React 19, hand-CSS in `app/globals.css` (Tailwind v4 preflight only). Markdown rendering via `react-markdown`.
- **Synthesis:** AI SDK v6 → Vercel AI Gateway → `anthropic/claude-sonnet-4-6`. Moderation uses `claude-haiku-4-5-20251001`.
- **NLA service:** Modal (`modal/penny_nla.py`), H100 GPU, transformers 4.56+.
- **Cache + rate limit:** Upstash Redis via the Vercel Marketplace.
- **Deployed:** Vercel.

## Local dev

```bash
cp .env.local.example .env.local      # fill in the four env blocks
pnpm install
pnpm dev                              # http://localhost:3000
```

If `MODAL_DECODE_URL` or `AI_GATEWAY_API_KEY` is unset, `/api/ask` returns 502 and the client falls back to a mock response so the UI still renders with placeholder text.

To deploy the Modal NLA service yourself, see [`modal/README.md`](./modal/README.md). You'll need a HuggingFace token with access to `google/gemma-3-12b-it` (gated — accept the license on the model page) and to `kitft/nla-gemma3-12b-L32-av`.

## Project structure

- `app/page.tsx` — chat shell, phase state machine (idle → loading → answered → spending → revealing → revealed), typewriter animation
- `app/api/ask/route.ts` — POST endpoint chaining cache lookup, rate-limit, moderation, Modal call, Claude synthesis
- `lib/modal-client.ts` — typed fetch wrapper for the Modal `/decode` endpoint
- `lib/cache.ts` — Upstash Redis get/set with versioned hashed keys
- `lib/ratelimit.ts` — tiered per-IP rate limiting (8/min · 40/hr · 100/day)
- `lib/moderation.ts` — Claude Haiku 4.5 binary classifier; rejects slurs, weapons/self-harm asks, prompt-injection
- `lib/types.ts` — shared `ThoughtResult` type (the UI's wire schema)
- `components/NeuralNetwork.tsx` — canvas viz with cascade-wave on activity
- `components/Penny.tsx` — 3D CSS coin: idle hover → drop → resting
- `modal/penny_nla.py` — the GPU service: model load, layer-32 forward hook, AV-batch verbalize
- `app/globals.css` — palette, typography, layout

## Honest framing

NLA decodings are interpretations of the model's residual stream — not literal transcripts of "thought." The verbalizer compresses a 3,840-dimensional vector into one English sentence, which inevitably loses information. The Claude synthesis layer can over-narrativize. The chips are condensed by Claude from the verbalizer's verbose explanations and inherit any biases of that pass.

The footer of the live site links the source paper, blog, and recipe so anyone curious can dig into what's real and what's aesthetic.

## Acknowledgments

This project is a portfolio piece built on top of substantial open work by others:

- **Anthropic** — the [Natural Language Autoencoders research](https://transformer-circuits.pub/2026/nla/index.html) ([blog](https://www.anthropic.com/research/natural-language-autoencoders) · [video](https://www.youtube.com/watch?v=j2knrqAzYVY)). The verbalizer/reconstructor mechanism, the L2-norm-then-injection-scale convention, and the prompt template format are all from this work.
- **kitft** — [`natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders) (Apache 2.0). The canonical inference recipe, the `nla_meta.yaml` sidecar schema, and the released [`nla-gemma3-12b-L32-av`](https://huggingface.co/kitft/nla-gemma3-12b-L32-av) checkpoint that this app uses at runtime. None of kitft's source code is vendored into this repo, but `modal/penny_nla.py` follows their documented pattern.
- **Google** — [Gemma 3](https://ai.google.dev/gemma) is the base model. Use of `google/gemma-3-12b-it` requires accepting the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) on Hugging Face. The AV checkpoint is also covered by the Gemma license because it is a fine-tune of Gemma weights.

This project is **not affiliated with Anthropic, Google, or the kitft author.** It is built for learning and demonstration.

## License

This project's original code is released under the [MIT License](./LICENSE). Upstream dependencies retain their own licenses (linked above).
