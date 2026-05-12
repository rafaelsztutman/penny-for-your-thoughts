# penny-nla — Modal NLA service

Runs Gemma-3-12B-IT plus the [`kitft/nla-gemma3-12b-L32-av`](https://huggingface.co/kitft/nla-gemma3-12b-L32-av) verbalizer on an H100. Exposes one HTTP endpoint that returns Gemma's answer to a question alongside per-token natural-language decodings of the residual stream at layer 32.

The Next.js app at the repo root (`/api/ask`) calls this endpoint, runs a Claude synthesis pass over the result, and renders the chips.

## Prerequisites

1. **Modal CLI**, installed and authenticated:

   ```bash
   pipx install modal
   modal token new
   modal token info       # confirm
   ```

2. **HuggingFace token** with read access to the gated `google/gemma-3-12b-it` model. (Accept the license on the model page first; then create a fine-grained token with "Read access to contents of all public gated repos you can access".) Expose it to Modal as a secret named `hf-token`:

   ```bash
   modal secret create hf-token HF_TOKEN=hf_yourTokenHere
   ```

## Deploy

```bash
modal deploy modal/penny_nla.py
```

The first deploy builds the image (~2 minutes for `torch` + `transformers`). Subsequent deploys are ~2 seconds since the image is cached.

The output prints a web endpoint URL like:

```
https://<workspace>--penny-nla-nla-decode.modal.run
```

Save it as `MODAL_DECODE_URL` in your Vercel project's environment variables.

## Test the endpoint

```bash
curl -X POST "$MODAL_DECODE_URL" \
  -H 'content-type: application/json' \
  -d '{"question":"Why do leaves change color in autumn?"}'
```

The first call cold-starts (~5 minutes the very first time, downloading both models to the persistent HF cache volume; ~30–40 seconds on subsequent cold starts). Calls within the 60-second `scaledown_window` after a previous request reuse the warm container.

## Request body

| field | type | default | meaning |
|---|---|---|---|
| `question` | string | — | The user's prompt sent to Gemma. Required. |
| `max_tokens` | int | 1024 | Cap on Gemma's response length. Hard ceiling 1024. |
| `target_chips` | int | 32 | Number of evenly-spaced positions to capture across the actual generated answer. Short answers get dense sampling; long answers, sparse. Hard ceiling 40. Set to `0` to fall back to legacy `every_nth` mode. |
| `every_nth` | int | 64 | Legacy fixed-stride sampling (only used if `target_chips=0`). |
| `av_max_new_tokens` | int | 80 | Max length of each AV verbalization. Higher = richer per-chip blobs at the cost of slower decode. Range 40–220. |

## Response shape

```json
{
  "response": "Leaves change color in autumn because…",
  "pre_response_thought": "preparing a biological explanation about chlorophyll",
  "tokens": [
    { "idx": 0,  "text": " green",      "thought": "chlorophyll concept; plant pigment" },
    { "idx": 8,  "text": " sunlight",   "thought": "photosynthesis input; energy capture" },
    ...
  ],
  "latency_ms": { "generate": 1830, "decode": 940 },
  "meta": {
    "base_model": "google/gemma-3-12b-it",
    "av_model": "kitft/nla-gemma3-12b-L32-av",
    "layer": 32
  }
}
```

`tokens` is in generation order. `pre_response_thought` is the verbalization of the residual at the final input position — what Gemma was "thinking about" the moment before generating its first answer token.

## Troubleshooting

- **CJK-flavoured output from the AV** — tokenizer drift or wrong injection scale. The load-time assertions in `_verbalize_batch` catch this for the canonical AV prompt; if you see CJK output anyway, double-check that the `nla_meta.yaml` sidecar matches the checkpoint version.
- **L2 norm of captured vector = 0** — wrong layer index or hook not firing. Check the hook target (`model.model.layers[layer_idx]` for Gemma-3).
- **Container OOM at load** — Gemma-3-12B + AV in bf16 ≈ ~50 GB. An H100 80GB fits both, but only just. Don't add a third model without checking memory.
- **First cold start > 5 minutes** — the HF cache volume is empty. Subsequent cold starts (cache populated) are ~30 seconds.
- **`<end_of_turn>` chips appearing in output** — should be filtered by both the in-loop stop condition and the post-generation special-token filter. If they reappear, check whether the model config's `generation_config.eos_token_id` includes the turn-end markers (it should be a list).

## Acknowledgments

The inference recipe (forward-hook layer extraction, `<INJECT>` injection-token swap, L2-normalize-and-rescale-to-injection-scale, sidecar-driven prompt template) is from [`kitft/natural_language_autoencoders`](https://github.com/kitft/natural_language_autoencoders) (Apache 2.0). This file is original code that follows that documented pattern; none of kitft's source is vendored.
