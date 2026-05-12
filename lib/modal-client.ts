import { z } from "zod";

const ModalTokenSchema = z.object({
  idx: z.number().nullable(),
  text: z.string(),
  thought: z.string(),
});

const ModalResponseSchema = z.object({
  response: z.string(),
  pre_response_thought: z.string().nullable(),
  tokens: z.array(ModalTokenSchema),
  latency_ms: z
    .object({ generate: z.number(), decode: z.number() })
    .optional(),
  meta: z.unknown().optional(),
});

export type ModalDecoding = z.infer<typeof ModalResponseSchema>;

export async function callModal(
  question: string,
  opts: {
    everyNth?: number;
    maxTokens?: number;
    avMaxNewTokens?: number;
    targetChips?: number;
    signal?: AbortSignal;
  } = {},
): Promise<ModalDecoding> {
  const url = process.env.MODAL_DECODE_URL;
  if (!url) throw new Error("MODAL_DECODE_URL not set");

  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      question,
      // target_chips: pick this many evenly-spaced capture positions across
      // the actual answer. Short answers get dense sampling (almost every
      // token), long answers get sparse — both end up at ~target_chips.
      // every_nth is the legacy fallback when target_chips is 0/unset.
      target_chips: opts.targetChips ?? 32,
      every_nth: opts.everyNth ?? 64,
      max_tokens: opts.maxTokens ?? 1024,
      av_max_new_tokens: opts.avMaxNewTokens ?? 80,
    }),
    signal: opts.signal,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`modal ${res.status}: ${body.slice(0, 200)}`);
  }
  const json = await res.json();
  return ModalResponseSchema.parse(json);
}
