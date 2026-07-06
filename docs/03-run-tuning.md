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
-ngl 99                       # full GPU offload — E4B fits entirely
-c 131072                     # full trained context — fits at ~65% VRAM (see below)
-fa on                        # flash attention: 2.7-4.4x prefill on gfx900 (see below)
-ctk q8_0 -ctv q8_0           # quantize KV cache (~half the size of f16)
-b 256 -ub 256                # measured prefill optimum on this stack (see below)
--temp 0.2 --top-p 0.95 --top-k 64   # sampling (see Sampling parameters)
--alias gemma4                # advertise a clean model id (instead of the long file path)
--host 0.0.0.0                # bind all interfaces so the LAN/opencode can reach it
--port 8080
```

### `-fa` syntax gotcha

This build requires a **value**: `-fa on` (or `off`/`auto`). Bare `-fa` fails because it swallows the next token as its value:

```
error: unknown value for --flash-attn: '-ctk'
```

Use **`-fa on`** explicitly — and you *do* want it on: it's a large speedup on this card (see below), not just a KV-cache convenience.

## VRAM / context math

KV cache grows linearly with context. The cost-per-token was **measured** on this exact setup (E4B, `q8_0` KV) by reading `radeontop` at several context sizes — see the method in [Detecting VRAM spill](#detecting-vram-spill).

> **Measured, not estimated:** the real cost is **~7.5 MiB per 1K tokens** — far cheaper than early guesses (~50 MiB/1K). Doubling context barely moves VRAM.

| `-c` value | VRAM under load (measured) | GTT (system RAM) | Spill? |
|---|---|---|---|
| 32768 | **4912 MB** | ~970 MB | No |
| 65536 | **5168 MB** (+256 MB) | ~972 MB | No |
| 131072 | **5628 MB** (+716 MB vs 32K) | ~1110 MB | No |

Per-token cost from the deltas: 32K→64K = ~7.8 MiB/1K, 64K→128K = ~7.0 MiB/1K → **~7.5 MiB/1K** average, rock-steady linear scaling.

**Takeaway:** even at the model's full trained context of **131072** (`n_ctx_train`), E4B sits at just **69% VRAM (5.6 GB of 8 GB)** with no spill. On this card, **128K is comfortably the practical max** — the limit is the model's useful context, not VRAM. That's why `-c 131072` is the recommended default.

### Tuning procedure

1. Start at the recommended value, watch `radeontop` during a real workload.
2. If VRAM stays well under ~7.5 GiB **and GTT stays flat** → you have headroom to spare (default already at full 128K context).
3. If it OOMs, maxes VRAM, or **GTT climbs** (see below) → drop `-c`, or `-b`.
4. Need to free VRAM? Use **`-ctk q4_0 -ctv q4_0`** (4-bit KV) to halve the cache, with a small quality tradeoff on very long contexts.

> Change **one lever at a time** and re-check.

## Prefill / batch-size tuning

`-b` (batch) and `-ub` (micro-batch) control how many prompt tokens are processed per step during **prefill** (prompt ingestion). They mainly affect how fast a prompt is *read*, not generation speed. For coding via opencode, where every request re-ingests a large context, prefill throughput matters a lot.

> **Methodology note:** the numbers below come from **`llama-bench`**, not ad-hoc `curl` timing. `llama-bench` warms up, repeats each test (`-r 5`), and reports mean ± stdev — so it avoids the prompt-cache trap (a re-sent identical prompt hits llama.cpp's KV cache and reports a fake ~29 t/s) and averages out transient noise. **On a Vega 64, pin the GPU clocks first** and keep the machine quiet — otherwise long tests (32K) show large error bars from P-state bouncing / thermal effects:
> ```bash
> echo manual | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
> echo high   | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
> # ...run llama-bench as your user (no sudo, so the HF cache is found)...
> echo auto   | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
> ```
> (Also stop any `radeontop` stream and close other GPU apps during timing. `llama-bench --prio 1..3` needs root; skip it — pinned clocks matter far more.)

### Micro-batch (`-ub`) sweep

Measured with `-b 256`, `-fa on`, `q8_0` KV (both K and V), `-r 5`, across three prompt lengths (t/s, higher is better):

| `-ub` | pp2048 | pp16384 | pp32768 |
|---|---|---|---|
| 128 | 648.65 | 535.62 | 467.47 |
| **256** | **665–676** | **562–567** | **453–479** |
| 512 | 623.86 | 540.15 | 472.29 |

**`-ub 256` is the sweet spot**: it beats 128 (~14% at 16K) and edges out 512, which starts to regress. Throughput rises to 256, then flattens/dips — past the optimum, larger micro-batches overflow the tiny cache and add scratch-buffer pressure. On this **cache-starved GCN5 die (4 MB L2, no L3)**, ~256 tokens per micro-batch is about the point where memory bandwidth saturates without blowing L2 locality. A good reminder to **measure, not assume "bigger = faster"** — a separate `-b 1024` bench showed `ub` *collapsing* at 1024 (255 t/s) purely from oversized batching.

> An informal `curl` test once suggested `ub=64` was faster; `llama-bench` did **not** confirm it — `ub=128` (64's neighbor) is measurably *below* the 256 peak. That earlier result was measurement noise, not a real effect. (The tempting "64 CUs → ub 64" mapping doesn't hold either: `-ub` counts *tokens*, not compute units, and the optimum is set by L2 locality, not CU count.)

Generation (`tg128`) is flat at **~66 t/s** regardless of `-ub` — micro-batch only affects prefill, as expected.

The original `-b 16 -ub 16` was a **crash workaround** for early RADV instability — once stable, it left ~2× prefill on the table.

VRAM impact of batch size is negligible — at `-c 131072` + `-b 256` + a 20K prompt, peak VRAM was **5258 MB (65%)** with GPU pinned and GTT flat (no spill).

> If you hit instability (crashes/hangs) on your driver stack, step back toward `-b 64` or `-b 16`. On this setup (Mesa RADV, CachyOS), `-b 256` was stable across repeated runs.

### Flash Attention on gfx900 — testing the folklore

A widely-repeated claim holds that **Flash Attention is slower on Vega 64 / gfx900** ("the compute units struggle with FA's memory access patterns"). **We tested it with `llama-bench` — and on this stack it is false.** FA is a large win at *every* context length, and the advantage **grows** with context:

| Test | FA **on** t/s | FA **off** t/s | FA speedup |
|---|---|---|---|
| pp2048 | 773.42 | 288.24 | **2.68×** |
| pp16384 | 643.92 | 186.22 | **3.46×** |
| pp32768 | 580.47 | 133.10 | **4.36×** |
| tg128 (generation) | 68.20 | 33.85 | **2.01×** |

*(Measured `-b 256 -ub 256`, `q8_0` K-cache; the FA-off leg used f16 V-cache. The gap is far too large for the V-cache format to matter.)*

Two takeaways:
1. **Keep `-fa on`.** It is one of the most impactful flags in the config — 2.7–4.4× prefill and ~2× generation. FA-off would be crippling at 128K.
2. **The claim is likely stale/misattributed.** llama.cpp's Vulkan FA shaders have improved, and current Mesa/RADV handles them well on gfx900. Also, **FA is what makes quantized (`q8_0`) KV efficient** — note generation *doubled* with FA on; the non-FA quantized-KV path falls off a cliff. Reports of "FA is slower" may have compared FA-off + f16 KV against something else.

> The lesson: **measure hardware folklore on your own card.** A claim that's true for one driver/model/context can be flatly wrong for yours — the only way to know is to benchmark it.

## Sampling parameters

Google's recommended sampling defaults for Gemma:

```bash
--temp 1.0 --top-p 0.95 --top-k 64
```

llama.cpp's built-in defaults (e.g. `temp 0.8`, `top-k 40`) are *not* what Gemma was calibrated for, so passing the official trio gives the intended behavior. Performance impact is effectively zero.

**For coding/agentic use via opencode**, lower the temperature for more deterministic output:

| Use case | Suggested |
|---|---|
| General chat (Google default) | `--temp 1.0 --top-p 0.95 --top-k 64` |
| **Coding via opencode** | `--temp 0.2 --top-p 0.95 --top-k 64` |

`--temp 0.0` gives fully greedy/deterministic output if you want maximum repeatability. Note: opencode may send its own per-request sampling params, which override these server-side defaults.

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

E4B is multimodal — the server loads an `mmproj-BF16.gguf` projector automatically, enabling **image** (and experimental **audio**) input. It adds ~0.5 GB VRAM.

**Should you disable it with `--no-mmproj` for coding?** On E4B, **no — keep it.** The projector only runs when you actually send an image; for pure text/coding it sits idle and costs **no tok/s**. With ~2.5 GB headroom even at 128K, the ~0.5 GB it occupies isn't needed elsewhere, so there's no runtime benefit to dropping it — keep it for image support. (Contrast the 12B above, where `--no-mmproj` *was* worth it purely to reclaim VRAM and reduce spill.) Harmless startup warnings about control tokens can be ignored.

Next: [04 — opencode integration](04-opencode.md)
