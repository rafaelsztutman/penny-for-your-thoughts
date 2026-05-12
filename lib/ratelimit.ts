import { Redis } from "@upstash/redis";

// Tiered limits — short-window protects against burst abuse, long-window caps
// total daily spend per IP. Both have to pass for a request to proceed.
const LIMITS = [
  { window: "minute", seconds: 60, max: 8 },
  { window: "hour", seconds: 60 * 60, max: 40 },
  { window: "day", seconds: 60 * 60 * 24, max: 100 },
] as const;

let _client: Redis | null = null;

function client(): Redis | null {
  if (_client) return _client;
  const url = process.env.KV_REST_API_URL;
  const token = process.env.KV_REST_API_TOKEN;
  if (!url || !token) return null;
  _client = new Redis({ url, token });
  return _client;
}

export type RateLimitResult = {
  allowed: boolean;
  limit?: number;
  remaining?: number;
  resetSeconds?: number;
  window?: string;
};

export async function rateLimit(ip: string): Promise<RateLimitResult> {
  const redis = client();
  // No Redis configured (local dev without KV vars) → fail open. We still
  // have the zod input validation + 500-char cap as a baseline.
  if (!redis) return { allowed: true };

  for (const { window, seconds, max } of LIMITS) {
    const key = `rl:${window}:${ip}`;
    const count = await redis.incr(key);
    if (count === 1) {
      await redis.expire(key, seconds);
    }
    if (count > max) {
      const ttl = await redis.ttl(key);
      return {
        allowed: false,
        limit: max,
        remaining: 0,
        resetSeconds: ttl > 0 ? ttl : seconds,
        window,
      };
    }
  }
  return { allowed: true };
}

export function clientIp(req: Request): string {
  const fwd = req.headers.get("x-forwarded-for");
  if (fwd) return fwd.split(",")[0]!.trim();
  return (
    req.headers.get("x-real-ip") ??
    req.headers.get("cf-connecting-ip") ??
    "unknown"
  );
}
