"""Penny for Your Thoughts — Modal NLA service.

Loads Gemma-3-12B-IT and the kitft AV verbalizer on an H100, answers a
question, captures residual-stream activations at the configured layer, and
returns per-token NLA decodings that the Next.js app turns into a "thoughts"
panel.

Deploy:
    modal deploy modal/penny_nla.py

Test:
    curl -X POST https://<your-deployment>.modal.run/decode \\
        -H 'content-type: application/json' \\
        -d '{"question":"Why do leaves change color?"}'

Tunables (request body):
    every_nth  — sample stride for response tokens (default 4)
    max_tokens — max response length in tokens (default 192)
"""

from __future__ import annotations

import modal

APP_NAME = "penny-nla"
GEMMA_MODEL = "google/gemma-3-12b-it"
AV_MODEL = "kitft/nla-gemma3-12b-L32-av"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        # Pin transformers to 4.x; 5.x has a Gemma3 internal that passes
        # BatchEncoding into nn.Embedding.forward and crashes. 4.56+ supports
        # Gemma-3 cleanly.
        "transformers>=4.56.0,<5.0",
        "accelerate>=0.34.0",
        "safetensors",
        "sentencepiece",
        "pyyaml",
        "numpy",
        "huggingface_hub",
        "fastapi[standard]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            # HF caches must be on the persistent Modal Volume; otherwise
            # every cold start re-downloads ~50GB of weights. These need to
            # be set as image env so they apply BEFORE huggingface_hub
            # imports read them.
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf/hub",
            "TRANSFORMERS_CACHE": "/cache/hf/hub",
        }
    )
)

app = modal.App(APP_NAME, image=image)

hf_cache = modal.Volume.from_name("penny-hf-cache", create_if_missing=True)


@app.cls(
    gpu="H100",
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/cache": hf_cache},
    scaledown_window=60,
    timeout=600,
)
@modal.concurrent(max_inputs=2)
class NLA:
    @modal.enter()
    def load(self) -> None:
        """Load Gemma + AV once per container. ~30-60s cold."""
        import os
        import torch
        import yaml
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.environ.get("HF_TOKEN")
        assert token, "HF_TOKEN missing — set the hf-token Modal secret"

        dtype = torch.bfloat16
        self.device = "cuda"

        # Base model — answers the question + provides residual activations.
        self.base_tok = AutoTokenizer.from_pretrained(GEMMA_MODEL, token=token)
        self.base = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL,
            torch_dtype=dtype,
            device_map=self.device,
            token=token,
            attn_implementation="eager",
        ).eval()

        # AV (verbalizer) — turns a residual vector into a text explanation.
        self.av_tok = AutoTokenizer.from_pretrained(AV_MODEL)
        self.av = AutoModelForCausalLM.from_pretrained(
            AV_MODEL,
            torch_dtype=dtype,
            device_map=self.device,
            attn_implementation="eager",
        ).eval()

        # Sidecar — extraction layer, injection scale, prompt template,
        # tokenizer landmarks. Validate aggressively; silent failures here
        # produce CJK-flavoured output instead of real explanations.
        meta_path = hf_hub_download(AV_MODEL, "nla_meta.yaml")
        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        assert meta["role"] == "av"
        assert meta["kind"] == "nla_model"
        self.d_model = int(meta["d_model"])
        self.layer_idx = int(meta["extraction_layer_index"])
        self.injection_scale = float(meta["extraction"]["injection_scale"])
        self.injection_char = meta["tokens"]["injection_char"]
        self.injection_token_id = int(meta["tokens"]["injection_token_id"])
        self.injection_left_id = int(meta["tokens"]["injection_left_neighbor_id"])
        self.injection_right_id = int(meta["tokens"]["injection_right_neighbor_id"])
        self.av_prompt = meta["prompt_templates"]["av"]

        # Verify the AV tokenizer matches the sidecar — the canonical "drift
        # catch" from kitft's recipe.
        live_inj = self.av_tok.encode(self.injection_char, add_special_tokens=False)
        assert live_inj == [self.injection_token_id], (
            f"AV tokenizer drift: {self.injection_char!r} → {live_inj}, "
            f"sidecar says [{self.injection_token_id}]"
        )

        # Tokenize the canonical AV prompt once and verify there is exactly
        # one injection site with the expected neighbors. This catches prompt-
        # template drift or chat-template changes at load time, so we don't
        # have to re-check on every request like kitft does (their RL setting
        # has user-controlled rollouts; ours doesn't).
        canonical = self.av_prompt.format(injection_char=self.injection_char)
        canonical_ids = self.av_tok.apply_chat_template(
            [{"role": "user", "content": canonical}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not isinstance(canonical_ids, list):
            canonical_ids = (
                canonical_ids["input_ids"][0].tolist()
                if hasattr(canonical_ids, "input_ids")
                else canonical_ids[0].tolist()
            )
        sites = [i for i, t in enumerate(canonical_ids) if t == self.injection_token_id]
        assert len(sites) == 1, (
            f"canonical AV prompt has {len(sites)} injection sites, expected 1"
        )
        p = sites[0]
        assert 0 < p < len(canonical_ids) - 1, "injection at sequence boundary"
        assert canonical_ids[p - 1] == self.injection_left_id, (
            f"left-neighbor drift: got {canonical_ids[p-1]} "
            f"expected {self.injection_left_id}"
        )
        assert canonical_ids[p + 1] == self.injection_right_id, (
            f"right-neighbor drift: got {canonical_ids[p+1]} "
            f"expected {self.injection_right_id}"
        )
        self._av_prompt_ids = canonical_ids
        self._av_inj_pos = p

        # Locate the transformer-block module list. Gemma-3 architecture
        # varies between checkpoints: text-only sometimes exposes
        # `model.layers`, multimodal-wrapped checkpoints hide it under
        # `model.language_model.layers`. Probe both, fall back to walking
        # children for any ModuleList of the right length.
        import torch.nn as nn

        def _find_layers(m):
            candidates = [
                lambda x: x.model.layers,
                lambda x: x.model.language_model.layers,
                lambda x: x.language_model.model.layers,
                lambda x: x.language_model.layers,
            ]
            for f in candidates:
                try:
                    layers = f(m)
                    if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                        return layers
                except AttributeError:
                    continue
            # Fallback: walk submodules for a ModuleList that matches
            # config.num_hidden_layers.
            target = getattr(m.config, "num_hidden_layers", None) or getattr(
                getattr(m.config, "text_config", None), "num_hidden_layers", None
            )
            for sub in m.modules():
                if (
                    isinstance(sub, nn.ModuleList)
                    and target is not None
                    and len(sub) == target
                ):
                    return sub
            raise RuntimeError(
                f"could not locate transformer layers in {type(m).__name__}"
            )

        self.layers = _find_layers(self.base)
        assert 0 <= self.layer_idx < len(self.layers), (
            f"layer {self.layer_idx} out of range for "
            f"{len(self.layers)}-layer base"
        )
        print(f"[NLA] base has {len(self.layers)} layers; hooking idx {self.layer_idx}")

        # AV input-embedding table — used for the input_embeds injection trick.
        self.av_embed = self.av.get_input_embeddings()

        print(
            f"[NLA] loaded base={GEMMA_MODEL} av={AV_MODEL} "
            f"d={self.d_model} layer={self.layer_idx} "
            f"scale={self.injection_scale}"
        )

    # ─── activation capture during generation ──────────────────────────────

    def _generate_with_capture(
        self,
        question: str,
        max_tokens: int,
        every_nth: int,
        target_chips: int | None = None,
    ):
        """Generate Gemma's answer, capturing residual at extraction_layer
        at the last input token (pre-response) and at one position per
        response token. After generation, downsamples to either
        `target_chips` evenly-spaced positions (preferred — adapts to
        actual answer length) or every Nth (legacy mode).

        Returns (response_text, captured) where captured is a list of dicts
        {token_text, position, vector (cpu fp32 tensor), is_pre_response}.
        """
        import torch

        # Build chat-formatted input. transformers 5.x returns BatchEncoding
        # from apply_chat_template(return_tensors="pt"); 4.x returns a raw
        # tensor. Normalize.
        messages = [{"role": "user", "content": question}]
        chat_out = self.base_tok.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if hasattr(chat_out, "input_ids"):
            input_ids = chat_out["input_ids"].to(self.device)
        else:
            input_ids = chat_out.to(self.device)

        captured = []
        # Forward hook on the configured block — grabs the residual stream
        # AFTER the block runs. We update `last_layer_out` every forward.
        layer_out_ref = {"x": None}

        def hook(_module, _inputs, output):
            # Gemma block returns a tuple; first element is hidden states.
            hidden = output[0] if isinstance(output, tuple) else output
            layer_out_ref["x"] = hidden.detach()

        handle = self.layers[self.layer_idx].register_forward_hook(hook)

        try:
            with torch.inference_mode():
                # ─── initial forward (the question) ──────────────────────
                out = self.base(input_ids=input_ids, use_cache=True)
                past = out.past_key_values
                last_logits = out.logits[:, -1, :]

                # pre-response vector = residual at the final input position
                pre_vec = layer_out_ref["x"][0, -1, :].clone().to(torch.float32).cpu()
                captured.append({
                    "token_text": "",
                    "is_pre_response": True,
                    "vector": pre_vec,
                })

                # ─── autoregressive decode ───────────────────────────────
                # Stop on ANY token the model itself considers a turn-ender.
                # For Gemma-3, generation_config.eos_token_id is a LIST that
                # includes both <eos> and <end_of_turn>; tokenizer.eos_token_id
                # alone misses <end_of_turn>, so without this we'd generate
                # (and capture) trailing turn-end markers as noise chips.
                stop_ids: set[int] = set()
                gen_eos = getattr(self.base.generation_config, "eos_token_id", None)
                if isinstance(gen_eos, (list, tuple, set)):
                    stop_ids.update(int(x) for x in gen_eos if x is not None)
                elif gen_eos is not None:
                    stop_ids.add(int(gen_eos))
                if self.base_tok.eos_token_id is not None:
                    stop_ids.add(int(self.base_tok.eos_token_id))
                # Also catch any token whose decoded form is the literal
                # <end_of_turn>/<start_of_turn> markup, in case the model
                # config doesn't list it.
                for special in ("<end_of_turn>", "<eos>"):
                    sid = self.base_tok.convert_tokens_to_ids(special)
                    if (
                        sid is not None
                        and sid != self.base_tok.unk_token_id
                    ):
                        stop_ids.add(int(sid))
                print(f"[NLA] stop_ids={sorted(stop_ids)}", flush=True)
                # Capture EVERY response token's residual; we'll downsample
                # after generation so short answers get dense sampling and
                # long answers get sparse sampling — both ending at ~target_chips.
                all_captured: list[dict] = []
                generated_ids: list[int] = []
                cur = torch.argmax(last_logits, dim=-1, keepdim=True)
                for step in range(max_tokens):
                    cur_id = int(cur.item())
                    if cur_id in stop_ids:
                        break
                    generated_ids.append(cur_id)

                    vec = (
                        layer_out_ref["x"][0, -1, :]
                        .clone()
                        .to(torch.float32)
                        .cpu()
                    )
                    tok_text = self.base_tok.decode(
                        [cur_id], skip_special_tokens=False
                    )
                    all_captured.append({
                        "token_text": tok_text,
                        "is_pre_response": False,
                        "vector": vec,
                        "position": step,
                    })

                    out = self.base(
                        input_ids=cur,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out.past_key_values
                    cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

                response_text = self.base_tok.decode(
                    generated_ids, skip_special_tokens=True
                )

                # Drop captures whose decoded text is special-token markup
                # before downsampling so we don't waste a chip slot on noise.
                def _is_special(s: str) -> bool:
                    t = s.strip()
                    return len(t) > 1 and t.startswith("<") and t.endswith(">")
                clean = [c for c in all_captured if not _is_special(c["token_text"])]

                # Downsample. target_chips wins over every_nth: pick that many
                # evenly-spaced positions across the actual answer length.
                if target_chips is not None and target_chips > 0:
                    n = len(clean)
                    if n <= target_chips:
                        sampled = clean
                    else:
                        # Linear-spaced indices [0, n-1] inclusive.
                        sampled = [
                            clean[round(i * (n - 1) / (target_chips - 1))]
                            for i in range(target_chips)
                        ]
                else:
                    sampled = [c for c in clean if c["position"] % every_nth == 0]

                captured.extend(sampled)
                print(
                    f"[NLA] generated={len(all_captured)} clean={len(clean)} "
                    f"sampled={len(sampled)}",
                    flush=True,
                )
        finally:
            handle.remove()

        return response_text, captured

    # ─── AV decode: vector → text explanation ──────────────────────────────

    def _verbalize_batch(self, vectors, av_max_new_tokens: int = 80):
        """Run AV on a batch of vectors, return one explanation per vector.

        Uses the input_embeds injection trick: tokenize the AV prompt once,
        find the position of the injection token, and at inference time
        swap in the L2-normalized × injection_scale vector at that position.
        """
        import re
        import torch

        if not vectors:
            return []

        # Canonical AV prompt is tokenized + neighbor-validated once at
        # @enter(); reuse here instead of re-tokenizing every request.
        prompt_ids = self._av_prompt_ids
        inj_pos = self._av_inj_pos

        ids_t = torch.tensor([prompt_ids], device=self.device)
        base_embeds = self.av_embed(ids_t)  # [1, L, d]
        L, D = base_embeds.shape[1], base_embeds.shape[2]
        N = len(vectors)

        with torch.inference_mode():
            # Stack into [N, L, d]; replace position inj_pos in each row
            # with that vector's scaled embedding.
            batch = base_embeds.expand(N, L, D).contiguous()
            for i, v in enumerate(vectors):
                v_dev = v.to(self.device, dtype=batch.dtype)
                norm = v_dev.norm()
                scaled = (
                    v_dev if norm.item() == 0 else (v_dev / norm * self.injection_scale)
                )
                batch[i, inj_pos, :] = scaled

            attention_mask = torch.ones((N, L), device=self.device, dtype=torch.long)
            gen = self.av.generate(
                inputs_embeds=batch,
                attention_mask=attention_mask,
                max_new_tokens=av_max_new_tokens,
                do_sample=False,
                pad_token_id=self.av_tok.pad_token_id or self.av_tok.eos_token_id,
            )
            texts = self.av_tok.batch_decode(gen, skip_special_tokens=True)

        outs = []
        for text in texts:
            m = re.search(
                r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL
            )
            if m:
                cleaned = m.group(1)
            else:
                cleaned = re.sub(
                    r"</?explanation>", "", text, flags=re.IGNORECASE
                )
            outs.append(cleaned.strip())
        return outs

    # ─── HTTP entrypoint ───────────────────────────────────────────────────

    @modal.fastapi_endpoint(method="POST", docs=True)
    def decode(self, item: dict):
        """Single-question pipeline: answer + per-token NLA decodings."""
        import time

        question = (item.get("question") or "").strip()
        if not question:
            return {"error": "missing question"}
        every_nth = int(item.get("every_nth", 64))
        max_tokens = int(item.get("max_tokens", 1024))
        # target_chips wins over every_nth: pick that many evenly-spaced
        # capture positions across the actual answer length, regardless of
        # how short or long the answer turned out to be. Short answers get
        # dense sampling (almost every token); long answers get sparse.
        # 0 (or omitted with every_nth set) = legacy fixed-stride mode.
        target_chips = int(item.get("target_chips", 0))
        target_chips = max(0, min(target_chips, 40))
        # Controls how much text each AV verbalization is allowed to produce.
        av_max_new_tokens = int(item.get("av_max_new_tokens", 80))
        av_max_new_tokens = max(40, min(av_max_new_tokens, 220))
        every_nth = max(1, min(every_nth, 96))
        # 1024 tokens leaves enough room for Gemma to actually finish its
        # structured explainers; combined with a sentence-boundary trim on
        # the Next.js side, answers feel complete instead of cut.
        max_tokens = max(32, min(max_tokens, 1024))

        t0 = time.time()
        response_text, captured = self._generate_with_capture(
            question,
            max_tokens=max_tokens,
            every_nth=every_nth,
            target_chips=target_chips if target_chips > 0 else None,
        )
        t_gen = time.time() - t0

        t1 = time.time()
        explanations = self._verbalize_batch(
            [c["vector"] for c in captured],
            av_max_new_tokens=av_max_new_tokens,
        )
        t_dec = time.time() - t1

        # Belt-and-suspenders: drop captures whose decoded text looks like a
        # special-token marker (<end_of_turn>, <eos>, <bos>, etc.). These
        # surface as noise chips in the UI even when the loop should have
        # broken before capturing them.
        def _is_special_marker(text: str) -> bool:
            t = text.strip()
            return len(t) > 1 and t.startswith("<") and t.endswith(">")

        tokens = []
        pre_response_thought = None
        for c, expl in zip(captured, explanations):
            if c["is_pre_response"]:
                pre_response_thought = expl
                continue
            if _is_special_marker(c["token_text"]):
                continue
            tokens.append({
                "idx": c.get("position"),
                "text": c["token_text"],
                "thought": expl,
            })

        return {
            "response": response_text,
            "pre_response_thought": pre_response_thought,
            "tokens": tokens,
            "latency_ms": {
                "generate": int(t_gen * 1000),
                "decode": int(t_dec * 1000),
            },
            "meta": {
                "base_model": GEMMA_MODEL,
                "av_model": AV_MODEL,
                "layer": self.layer_idx,
            },
        }
