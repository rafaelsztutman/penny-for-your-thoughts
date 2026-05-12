import { createHash } from "node:crypto";
import { Redis } from "@upstash/redis";

// Bumping this invalidates all prior cache entries — do it whenever the
// shape of ThoughtResult changes OR when the prompts that produce it (Modal
// max_tokens, synthesis system prompt) change enough that cached entries
// would mislead a viewer.
const SCHEMA_VERSION = "v9";
const TTL_SECONDS = 60 * 60 * 24 * 365; // 1 year

let _client: Redis | null = null;

function client(): Redis | null {
  if (_client) return _client;
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) return null;
  _client = new Redis({ url, token });
  return _client;
}

function normalize(question: string): string {
  return question
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[?!.,;:'"]+$/g, "");
}

export function cacheKey(question: string): string {
  const norm = normalize(question);
  const hash = createHash("sha256").update(norm).digest("hex").slice(0, 24);
  return `thought:${SCHEMA_VERSION}:${hash}`;
}

export async function cacheGet<T>(key: string): Promise<T | null> {
  const c = client();
  if (!c) return null;
  try {
    const v = await c.get<T>(key);
    return v ?? null;
  } catch (err) {
    console.warn("cacheGet failed", err);
    return null;
  }
}

export async function cacheSet<T>(key: string, value: T): Promise<void> {
  const c = client();
  if (!c) return;
  try {
    await c.set(key, value, { ex: TTL_SECONDS });
  } catch (err) {
    console.warn("cacheSet failed", err);
  }
}
