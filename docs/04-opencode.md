# 04 — Connecting opencode to llama-server

The cleanest path is to point opencode **directly** at your tuned `llama-server` over the LAN — bypassing Unsloth Studio's API layer entirely.

## Why direct, not through Studio?

Studio exposes its own OpenAI-compatible API (default `:8888`), but it has a **separate internal inference engine**. Connecting an external llama.cpp under *Settings → Connections* does **not** make Studio's `/v1` endpoint serve that server — calls return:

```
No model loaded. Call POST /inference/load first.
```

That means Studio expects you to load a model into *its* engine. To guarantee opencode uses **your** Vulkan-tuned server (with `-fa on`, 32K context, etc.), connect to `llama-server` directly.

| Path | opencode points at | Result |
|---|---|---|
| Through Studio (`:8888`) | Studio's engine | Needs `/inference/load`; may not use your tuned server |
| **Direct (`:8080`)** | your `llama-server` | **Guaranteed** to use your tuned flags |

## 1. Make sure the server is LAN-reachable

Launch with `--host 0.0.0.0` (see [`run_server.sh`](../run_server.sh)). Then, from the opencode machine:

```bash
curl http://<SERVER_IP>:8080/v1/models
```

You should get JSON back listing the model (and `"n_ctx": 32768`, `"owned_by": "llamacpp"`).

- Connection refused → server still bound to localhost, or a firewall on `:8080`.
- Works locally but not from another box → CachyOS firewall.

> Replace `<SERVER_IP>` with your server's LAN address.

## 2. opencode.jsonc

llama.cpp is OpenAI-compatible, so use the openai-compatible adapter and override `baseURL`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Gemma 4 (Vulkan)",
      "options": {
        "baseURL": "http://<SERVER_IP>:8080/v1"
      },
      "models": {
        "gemma4": {
          "name": "Gemma 4 E4B QAT"
        }
      }
    }
  }
}
```

| Field | Value | Why |
|---|---|---|
| provider key | `llamacpp` | Arbitrary id you reference |
| `npm` | `@ai-sdk/openai-compatible` | Adapter for OpenAI-compatible servers |
| `options.baseURL` | `http://<SERVER_IP>:8080/v1` | Your server (`/v1` endpoint) |
| `models` key | `gemma4` | Model **id** opencode sends — must match the server's `--alias gemma4` |

Reference the model in opencode as **`llamacpp/gemma4`** (provider key + model key).

### Model id must match

opencode sends the `models` key (`gemma4`) as the requested model. Launch the server with **`--alias gemma4`** so it advertises that exact id. Otherwise the model id is the full GGUF filesystem path and you'd have to use that verbatim.

### API key

Only if you launched with `--api-key`, add it under `options`:

```jsonc
"options": {
  "baseURL": "http://<SERVER_IP>:8080/v1",
  "apiKey": "local-key"
}
```

llama.cpp accepts any non-empty key if you didn't set one.

## 3. Smoke test

```bash
curl http://<SERVER_IP>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Say hi in 3 words"}]}'
```

A coherent reply = the integration is live. Watch `radeontop` to confirm the Vega is busy (~5–6 GiB VRAM) — proof it's running on Vulkan, not CPU.

> opencode's config schema can shift between versions; if your version nests provider config differently, adapt the shape above accordingly.
