# gemma4-vulkan-cachyos

> **💡 Not on a Vega 64?** This setup works on **any GPU with a working Vulkan driver** — most usefully, **newer AMD cards** (RDNA / RX 6000 / 7000 series). The Vega 64 (gfx900) is just *this* box; nothing here is Vega-specific except the tuning numbers.

Running **Google Gemma 4 (E4B QAT)** locally on a **low-end AMD Vega 64 (gfx900)** under **CachyOS**, using **llama.cpp with the Vulkan backend** — and serving it to [opencode](https://opencode.ai).

This repo documents a full, hard-won setup: building llama.cpp for Vulkan on CachyOS, enabling Vulkan in Unsloth Studio (official on recent versions; a manual binary swap on older ones), tuning the run parameters for an 8 GB card, and wiring it into opencode.

> **Note:** As of **Unsloth v0.1.50-beta**, Studio has official Vulkan support — no binary swap needed. See [docs/02](docs/02-studio-swap.md).

## Why Vulkan instead of ROCm?

The Vega 64 (gfx900) lost official **ROCm** support after ROCm 5.6, which breaks the PyTorch/bitsandbytes/Triton stack Unsloth needs for **fine-tuning**. The **Vulkan** backend of llama.cpp still runs it well for **inference**.

> **Important:** Vulkan = **inference only**. Fine-tuning/training still requires ROCm and is **not** possible on this card locally. Train in the cloud, run the GGUF here.

## Hardware / software target

| Component | Value |
|---|---|
| GPU | AMD Radeon RX Vega 64 (gfx900), 8 GB |
| Driver | Mesa **RADV** (Vulkan) |
| OS | CachyOS (Arch-based) |
| Engine | llama.cpp, `-DGGML_VULKAN=ON` |
| Model | `unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL` (~4.2 GB) |
| Client | opencode via OpenAI-compatible API |

## Quick start

```bash
# 1. Build llama.cpp with Vulkan (see docs/01-build-vulkan.md)
# 2. Launch the tuned server:
./run_server.sh
# 3. Point opencode at it (see docs/04-opencode.md)
```

## Documentation

| Guide | What it covers |
|---|---|
| [docs/01-build-vulkan.md](docs/01-build-vulkan.md) | CachyOS dependencies (RADV, Vulkan headers, SPIRV-Tools) and building llama.cpp with Vulkan |
| [docs/02-studio-swap.md](docs/02-studio-swap.md) | **(Optional — only if you use Unsloth Studio)** Enabling Vulkan in Studio — official `UNSLOTH_FORCE_VULKAN` support on recent versions, with the legacy binary swap as a fallback |
| [docs/03-run-tuning.md](docs/03-run-tuning.md) | Launch parameters, the E4B QAT profile, and VRAM/context tuning math |
| [docs/04-opencode.md](docs/04-opencode.md) | Connecting opencode directly to llama-server over the LAN |

## The validated launch command

```bash
./llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL \
  --alias gemma4 \
  -ngl 99 -c 131072 -fa on \
  -ctk q8_0 -ctv q8_0 \
  -b 256 -ub 256 \
  --temp 0.2 --top-p 0.95 --top-k 64 \
  --host 0.0.0.0 --port 8080
```

See [`run_server.sh`](run_server.sh) for the runnable version, and [docs/03](docs/03-run-tuning.md) for how each flag was tuned (measured ~2.17× prefill at `-b 256`, full 128K context at ~65% VRAM).

## Status

- [x] RADV / Vulkan detected
- [x] llama.cpp built with `-DGGML_VULKAN=ON`
- [x] Full GPU offload confirmed (`Vulkan0` buffer in VRAM)
- [x] E4B QAT model running, multimodal (`mmproj`) loaded
- [x] 128K context, ~5.3 GB VRAM under load (radeontop, ~65% — no spill)
- [x] Prefill tuned: `-b 256` → ~336 tok/s (~2.17× vs `-b 16`)
- [x] opencode connected directly over `:8080`
