#!/usr/bin/env bash
#
# Launch Gemma 4 E4B QAT on a Vega 64 via llama.cpp (Vulkan backend),
# exposed over the LAN for opencode.
#
# See docs/03-run-tuning.md for the meaning of each flag and the
# VRAM/context tuning math.

set -euo pipefail

# --- config -----------------------------------------------------------------
# Path to your Vulkan-built llama-server (from docs/01-build-vulkan.md).
# Adjust if yours lives elsewhere.
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"

# Model (pulled from Hugging Face by -hf). Override MODEL to use a local path.
MODEL="${MODEL:-unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL}"

# Context window. 131072 (full trained context) fits at ~65% VRAM on 8 GB
# with q8_0 KV — measured, no spill. Drop to 32768 if you want more headroom.
CTX="${CTX:-131072}"

# Bind address / port. 0.0.0.0 = reachable on the LAN (needed for opencode
# running on another machine). Use 127.0.0.1 to restrict to localhost.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
# ---------------------------------------------------------------------------

exec "$LLAMA_SERVER" \
  -hf "$MODEL" \
  --alias gemma4 \
  -ngl 99 \
  -c "$CTX" \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -b 256 -ub 256 \
  --temp 0.2 --top-p 0.95 --top-k 64 \
  --host "$HOST" --port "$PORT"
