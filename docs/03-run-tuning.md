# 03 — Run parameters & tuning (E4B QAT on 8 GB)

How the launch flags were chosen for Gemma 4 **E4B QAT** on a Vega 64.

## Why E4B QAT?

| Option | Weights | Fully fits 8 GB? | Notes |
|---|---|---|---|
| 12B Q4_K_M | ~7 GB | No — KV cache spills, forces partial offload | Best quality, but slow (CPU spill) |
| **E4B QAT (UD-Q4_K_XL)** | **~4.2 GB** | **Yes**, with room for big context | QAT keeps quality high at 4-bit |

**QAT (Quantization-Aware Training)** simulates quantization *during* training, so the 4-bit model behaves close to full precision. It does **not** shrink VRAM below the bit-width — it makes a small, fully-in-VRAM config actually worth running. The **UD-Q4_K_XL** is Unsloth's Dynamic quant, a step above generic Q4_K_M.

## The flags

```bash
-ngl 99              # full GPU offload — E4B fits entirely
-c 32768             # context window (see tuning below)
-fa on               # flash attention: keeps KV cache compact + stable on RADV
-ctk q8_0 -ctv q8_0  # quantize KV cache (~half the size of f16)
-b 16 -ub 16         # small batch / micro-batch — crash mitigation on this stack
--alias gemma4       # advertise a clean model id (instead of the long file path)
--host 0.0.0.0       # bind all interfaces so the LAN/opencode can reach it
--port 8080
```

### `-fa` syntax gotcha

This build requires a **value**: `-fa on` (or `off`/`auto`). Bare `-fa` fails because it swallows the next token as its value:

```
error: unknown value for --flash-attn: '-ctk'
```

Use **`-fa on`** explicitly — you want flash attention on for the KV-cache savings, not left to `auto`.

## VRAM / context math

KV cache grows linearly with context. With **q8_0** KV quant on E4B, the cache costs roughly **~50 MiB per 1K tokens**. Idle model load is ~4.4 GiB (weights + multimodal `mmproj`).

| `-c` value | Approx KV cache | Total VRAM | Fits 8 GB? |
|---|---|---|---|
| 8192 | ~0.4 GiB | ~4.4 GiB | Yes (was hitting agent compaction) |
| 16384 | ~0.8 GiB | ~5.2 GiB | Yes, comfortable |
| **32768** | ~1.6 GiB | **~6.0 GiB** | **Yes — recommended** |
| 65536 | ~3.2 GiB | ~7.6 GiB | Tight / risky |

The model was trained at **131072** context (`n_ctx_train`), so quality holds at long context — the only limit is VRAM.

### Tuning procedure

1. Start at the recommended value, watch `radeontop` during a real workload.
2. If VRAM stays well under ~7.5 GiB → raise `-c` one step (49152, 65536).
3. If it OOMs or maxes → drop `-c`, or `-b 8`.
4. Need more context but 64K is tight? Use **`-ctk q4_0 -ctv q4_0`** (4-bit KV) to halve the cache again, with a small quality tradeoff on very long contexts.

> Change **one lever at a time** and re-check.

## Multimodal note

E4B is multimodal — the server loads an `mmproj-BF16.gguf` projector automatically, enabling **image** (and experimental **audio**) input. It adds a small amount of VRAM. Harmless startup warnings about `</s>` / `<|tool_response>` token types are just llama.cpp auto-correcting token metadata — not errors.

Next: [04 — opencode integration](04-opencode.md)
