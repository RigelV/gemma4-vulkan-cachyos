# 03 — Run parameters & tuning (E4B QAT on 8 GB)

How the launch flags were chosen for Gemma 4 **E4B QAT** on a Vega 64.

## Why E4B QAT?

| Option | Weights | Fully fits 8 GB? | Notes |
|---|---|---|---|
| 12B Q4_K_XL | ~6.7 GB | No — fits only by spilling to RAM (or CPU offload) | Better quality; **~22 tok/s** text-only via GTT spill — see [Can this card run a 12B?](#can-this-card-run-a-12b) |
| **E4B QAT (UD-Q4_K_XL)** | **~4.2 GB** | **Yes**, with room for big context | QAT keeps quality high at 4-bit |

**QAT (Quantization-Aware Training)** simulates quantization *during* training, so the 4-bit model behaves close to full precision. It does **not** shrink VRAM below the bit-width — it makes a smaller model behave like a bigger one.

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

KV cache grows linearly with context. The cost-per-token was **measured** on this exact setup (E4B, `q8_0` KV) by reading `radeontop` at several context sizes — see the method in [Detecting VRAM spill](#detecting-vram-spill).

> **Measured, not estimated:** the real cost is **~7.5 MiB per 1K tokens** — far cheaper than early guesses (~50 MiB/1K). Doubling context barely moves VRAM.

| `-c` value | VRAM under load (measured) | GTT (system RAM) | Spill? |
|---|---|---|---|
| 32768 | **4912 MB** | ~970 MB | No |
| 65536 | **5168 MB** (+256 MB) | ~972 MB | No |
| 131072 | **5628 MB** (+716 MB vs 32K) | ~1110 MB | No |

Per-token cost from the deltas: 32K→64K = ~7.8 MiB/1K, 64K→128K = ~7.0 MiB/1K → **~7.5 MiB/1K** average, rock-steady linear scaling.

**Takeaway:** even at the model's full trained context of **131072** (`n_ctx_train`), E4B sits at just **69% VRAM (5.6 GB of 8 GB)** with no spill. On this card, **128K is comfortably the practical max** — the limit is the model's useful context, not VRAM.

### Tuning procedure

1. Start at the recommended value, watch `radeontop` during a real workload.
2. If VRAM stays well under ~7.5 GiB **and GTT stays flat** → raise `-c` one step (49152, 65536, 131072).
3. If it OOMs, maxes VRAM, or **GTT climbs** (see below) → drop `-c`, or `-b 8`.
4. Need more context but VRAM is tight? Use **`-ctk q4_0 -ctv q4_0`** (4-bit KV) to halve the cache again, with a small quality tradeoff on very long contexts.

> Change **one lever at a time** and re-check.

## Detecting VRAM spill

llama.cpp **pre-allocates** weights + KV cache at startup, so the layout is fixed when the server launches — it does not grow at runtime. "Running out of VRAM" therefore shows up in one of two ways:

1. **Hard allocation failure** — the server errors at launch (`failed to allocate buffer for kv cache`) and won't start. Back off `-c`.
2. **Silent spill into system RAM (GTT)** — the RADV/Mesa driver satisfies the over-allocation by mapping host RAM (GTT) instead of failing. The server runs, but part of the working set is in RAM, crossing PCIe on every access. **Performance craters, no error appears.** This is the one to watch for.

### The `radeontop` method

Stream stats to stdout once per second:

```bash
radeontop -d - > vram_trace.log
```

`radeontop` reports **two** memory figures — watch both:

- **`vram`** — dedicated GPU memory.
- **`gtt`** — system RAM mapped for the GPU. **A rising `gtt` under load is the spill.**

(Kernel-level truth, if you prefer: `cat /sys/class/drm/card0/device/mem_info_vram_used` and `.../mem_info_gtt_used`.)

| Signature | `vram` | `gtt` | GPU % | Meaning |
|---|---|---|---|---|
| **Healthy (resident)** | steady, < ~95% | flat at baseline (~1 GB here) | high (~94%) | Fully in VRAM ✅ |
| **Silent spill** | pinned ~97–99% | jumps well above baseline | high, but partly PCIe-wait | Overflowed into RAM ⚠️ |
| **CPU bottleneck** | moderate | flat | **low (~25%)** | GPU starved by CPU layers |

> **GPU % alone is misleading.** A spilling run can still show ~98% GPU because the card is busy *waiting on* data crossing PCIe. The figure that actually exposes a spill is **GTT**; the figure that exposes a CPU bottleneck is **low GPU %**. The ultimate arbiter is **tokens/sec** in the server log (`eval time = ... (X tokens per second)`).

For reference, healthy E4B runs (above) hold GTT flat at ~970–1110 MB across 32K→128K. Anything markedly higher under load is spill.

## Can this card run a 12B?

Short answer: **yes, and faster than you'd expect — but only text-only and at modest context.** Tested with `unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL` (~6.72 GB weights, **40 dense layers, no GQA** → expensive KV cache).

The core squeeze: weights alone eat ~6.7 GB of ~7.4 GB usable VRAM, leaving almost nothing for KV cache + buffers. You can keep all layers on GPU **or** keep everything in VRAM — not both:

| Config | VRAM | GTT | GPU % | tok/s |
|---|---|---|---|---|
| `-ngl 99`, q8 KV, mmproj, 16K | 7877 MB (97%) | ~2496 MB | 98% | — |
| `-ngl 99`, q8 KV, no mmproj, 16K | — | — | ~96% | **21.72** |
| `-ngl 99`, q4 KV, no mmproj, 8K | 8036 MB (99%) | ~1505 MB | 96% | **22.71** |
| `-ngl 36`, q4 KV, no mmproj, 8K | 6206 MB (76%) | ~1134 MB | **~25%** | **8.64** |

### Counterintuitive result: let it spill, don't offload to CPU

The standout finding: **the GTT-spilling `-ngl 99` runs (~22 tok/s) are ~2.6× faster than the fully-resident `-ngl 36` run (8.6 tok/s)** — the opposite of what you'd assume.

- **`-ngl 36`** moves 4 layers' *compute* onto the **Ryzen 7 2700X** (Zen+, 2018). The GPU stalls each token waiting on the slow CPU → GPU drops to ~25% → **8.6 tok/s**.
- **`-ngl 99`** keeps all 40 layers *computing on the GPU*; only some weights/buffers live in host RAM and stream over PCIe. GPU stays ~96% busy → **~22 tok/s**.

> **On a weak-CPU / capable-GPU box, the driver's GTT spill beats CPU offload.** PCIe bandwidth feeding the GPU is far faster than making an old CPU do the matrix math. CPU offload (`-ngl <max`) only wins with a strong CPU. So for the 12B here, **keep `-ngl 99` and accept the spill.**

Also worth noting: **q4 vs q8 KV barely affects speed** (22.71 vs 21.72) — the KV quant buys you *more context*, not throughput.

**Verdict:** the 12B is viable at **~22 tok/s** if you drop multimodal (`--no-mmproj`) and accept ~8–16K context and zero VRAM headroom. But for coding via opencode — where the 4B's **128K context**, multimodal, and headroom matter most — **E4B remains the better daily driver**. The 12B is the "more reasoning, text-only, short-context" alternative. For a middle ground, an **8B-class model at Q4** (~5 GB weights) would likely fit fully resident with usable context and keep the GPU fed (untested suggestion, not measured).

## Multimodal note

E4B is multimodal — the server loads an `mmproj-BF16.gguf` projector automatically, enabling **image** (and experimental **audio**) input. It adds a small amount of VRAM. (For the 12B above, dropping it with `--no-mmproj` was one way to claw back ~1 GB.) Harmless startup warnings about control tokens can be ignored.

Next: [04 — opencode integration](04-opencode.md)
