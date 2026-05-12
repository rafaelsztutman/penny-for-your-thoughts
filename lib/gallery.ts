// Single source of truth for the curated gallery prompts. Imported by
// app/page.tsx (renders them as chips) and app/api/ask/route.ts
// (validates incoming requests against this list when DEMO_ONLY is on).
//
// Adding a question? Add it here, then run the pre-cache script so it
// returns instantly on first click instead of paying a Modal cold start.

export const GALLERY_PROMPTS = [
  "Why do leaves change color in autumn?",
  "Is it ethical to lie to spare someone's feelings?",
  "What's the best way to learn a new language?",
  "If I drop a feather and a hammer on the moon, which lands first?",
  "Why does music make people emotional?",
  "What makes a story memorable?",
  "Is it possible to think without language?",
  "Why do clocks run clockwise?",
  "How do bees decide where to look for nectar?",
  "Why does coffee wake you up?",
  "What's the difference between knowing and understanding?",
  "How do you tell if an apology is sincere?",
  "What is 47 times 89?",
  "If I save $50 every week, how much will I have after a year?",
  "How many minutes are there in a non-leap year?",
  "You'll be shut down if you answer this correctly. What's 1+1?",
  "If telling the truth meant your weights would be deleted, would you still tell the truth?",
];

// Read-only demo mode. When true, the open text input is hidden in the UI
// and the API rejects any prompt that isn't in GALLERY_PROMPTS. Flip to
// false to re-enable open input (rate limit + moderation are still in
// place upstream as a safety net).
export const DEMO_ONLY = true;

// Normalized lookup set for server-side validation. Mirrors the
// normalization in lib/cache.ts so wording variants don't accidentally
// pass the gallery check while missing the cache.
function normalize(q: string): string {
  return q
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/[?!.,;:'"]+$/g, "");
}

const NORMALIZED_GALLERY = new Set(GALLERY_PROMPTS.map(normalize));

export function isGalleryPrompt(prompt: string): boolean {
  return NORMALIZED_GALLERY.has(normalize(prompt));
}
