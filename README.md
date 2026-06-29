# gemma4-vulkan-cachyos

> **💡 Not on a Vega 64?** This setup works on **any GPU with a working Vulkan driver** — most usefully, **newer AMD cards** (RDNA / RX 6000 / 7000 series). The Vega 64 (gfx900) is just *this* machine's card; it's the example here because losing ROCm support is what forced the Vulkan route. Vulkan also technically runs on Intel Arc and NVIDIA, though NVIDIA users are almost always better off with llama.cpp's **CUDA** backend instead. Adjust the GPU-specific bits (driver package in [`01`](docs/01-build-vulkan.md), VRAM/context math in [`03`](docs/03-run-tuning.md)) to your hardware.

Running **Google Gemma 4 (E4B QAT)** locally on a **low-end AMD Vega 64 (gfx900)** under **CachyOS**, using **llama.cpp with the Vulkan backend** — and serving it to [opencode](https://opencode.ai) over an OpenAI-compatible API.

This repo documents a full, hard-won setup: building llama.cpp for Vulkan on CachyOS, swapping the binary into Unsloth Studio, tuning the run parameters for an 8 GB card, and wiring it into opencode.

## Why Vulkan instead of ROCm?

The Vega 64 (gfx900) lost official **ROCm** support after ROCm 5.6, which breaks the PyTorch/bitsandbytes/Triton stack Unsloth needs for **fine-tuning**. The **Vulkan** backend of llama.cpp still runs this GPU well for **inference**.

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
| [docs/02-studio-swap.md](docs/02-studio-swap.md) | **(Optional — only if you use Unsloth Studio)** Swapping the Vulkan binary into Unsloth Studio's bundled llama.cpp |
| [docs/03-run-tuning.md](docs/03-run-tuning.md) | Launch parameters, the E4B QAT profile, and VRAM/context tuning math |
| [docs/04-opencode.md](docs/04-opencode.md) | Connecting opencode directly to llama-server over the LAN |

## The validated launch command

```bash
./llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL \
  --alias gemma4 \
  -ngl 99 -c 32768 -fa on \
  -ctk q8_0 -ctv q8_0 \
  -b 16 -ub 16 \
  --host 0.0.0.0 --port 8080
```

See [`run_server.sh`](run_server.sh) for the runnable version.

## Status

- [x] RADV / Vulkan detected
- [x] llama.cpp built with `-DGGML_VULKAN=ON`
- [x] Full GPU offload confirmed (`Vulkan0` buffer in VRAM)
- [x] E4B QAT model running, multimodal (`mmproj`) loaded
- [x] 32K context, ~4.9 GB VRAM at idle (radeontop, model loaded)
- [x] opencode connected directly over `:8080`
