# 03 — Run parameters & tuning (E4B QAT on 8 GB)

How the launch flags were chosen for Gemma 4 **E4B QAT** on a Vega 64.

> **`llama-bench` numbers are a no-overhead ceiling.** Every `tok/s` in this doc measured
> with **`llama-bench`** (the 4B's ~66 tg, the prefill tables) excludes the HTTP/JSON,
> prompt assembly, and streaming cost of a live `llama-server` + opencode — so **real
> served throughput is somewhat lower.** At the 4B's high tok/s the fixed per-token
> overhead takes a larger relative bite; for the 12B at 8K depth the two happen to
> converge within ~1% (see [Can this card run a 12B?](#can-this-card-run-a-12b)). Also
> note **generation speed depends on how full the KV cache is** — always benchmark at
> realistic depth (`llama-bench -d <ctx>`), since an empty-cache run overstates tok/s.

## Why E4B QAT?

| Option | Weights | Fully fits 8 GB? | Notes |
|---|---|---|---|
| 12B Q4_K_XL | ~6.7 GB | No — fits only by spilling to RAM (or CPU offload) | Better quality; **~23 tok/s** at 8K, text-only via GTT spill — see [Can this card run a 12B?](#can-this-card-run-a-12b) |
| **E4B QAT (UD-Q4_K_XL)** | **~4.2 GB** | **Yes**, with room for big context | QAT keeps quality high at 4-bit |

**QAT (Quantization-Aware Training)** simulates quantization *during* training, so the 4-bit model behaves close to full precision. It does **not** shrink VRAM below the bit-width — it makes a smaller quant *behave* like a bigger one.

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

**Takeaway:** even at the model's full trained context of **131072** (`n_ctx_train`), E4B sits at just **69% VRAM (5.6 GB of 8 GB)** with no spill. On this card, **128K is comfortably the practical default.**

### Tuning procedure

1. Start at the recommended value, watch `radeontop` during a real workload.
2. If VRAM stays well under ~7.5 GiB **and GTT stays flat** → you have headroom to spare (default already at full 128K context).
3. If it OOMs, maxes VRAM, or **GTT climbs** (see below) → drop `-c`, or `-b`.
4. Need to free VRAM? Use **`-ctk q4_0 -ctv q4_0`** (4-bit KV) to halve the cache, with a small quality tradeoff on very long contexts.

> Change **one lever at a time** and re-check.

## Prefill / batch-size tuning

`-b` (batch) and `-ub` (micro-batch) control how many prompt tokens are processed per step during **prefill** (prompt ingestion). They mainly affect how fast a prompt is *read*, not generation speed.

> **Methodology note:** the numbers below come from **`llama-bench`**, not ad-hoc `curl` timing. `llama-bench` warms up, repeats each test (`-r 5`), and reports mean ± stdev — so it avoids the cold-cache and single-shot noise that made an earlier `curl` result misleading. Pin the GPU clocks first so the run isn't muddied by P-state bouncing:
> ```bash
> echo manual | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
> echo high   | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
> # ...run llama-bench as your user (no sudo, so the HF cache is found)...
> echo auto   | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
> ```
> (Also stop any `radeontop` stream and close other GPU apps during timing. `llama-bench --prio 1..3` needs root; skip it — pinned clocks matter far more.)
>
> **Card number:** the Vega enumerates as **`card1`** here (`DRIVER=amdgpu`), but the number is **not stable** across boots/kernels. Resolve it dynamically instead of hardcoding:
> ```bash
> for c in /sys/class/drm/card[0-9]*; do
>   grep -ql amdgpu "$c/device/uevent" && [ -e "$c/device/pp_dpm_mclk" ] && echo "${c##*/}"
> done
> ```

### Micro-batch (`-ub`) sweep

Measured with `-b 256`, `-fa on`, `q8_0` KV (both K and V), `-r 5`, across three prompt lengths (t/s, higher is better):

| `-ub` | pp2048 | pp16384 | pp32768 |
|---|---|---|---|
| 128 | 648.65 | 535.62 | 467.47 |
| **256** | **665–676** | **562–567** | **453–479** |
| 512 | 623.86 | 540.15 | 472.29 |

**`-ub 256` is the sweet spot**: it beats 128 (~14% at 16K) and edges out 512, which starts to regress. Throughput rises to 256, then flattens/dips — past the optimum, larger micro-batches overflow the tiny cache and stall on memory.

> **Why micro-batch size matters more on Vega than on RDNA3 — the architecture**
>
> The Vega 64 (GCN5, gfx900) has an unusual memory hierarchy for its era, and it's the
> key to the `-ub` result:
>
> - **HBM2, 2048-bit bus, 484 GB/s.** Two HBM2 stacks × 1024-bit = 2048-bit total
>   (*not* the 4096-bit of the Radeon VII / MI-series), at 945 MHz effective. 484 GB/s
>   is a lot of raw bandwidth — but it only pays off if you keep the bus *busy*.
> - **Only 4 MB L2, and no L3 / Infinity Cache at all.** This is the crucial difference.
>   A modern RDNA3 card (e.g. RX 7900 XTX) pairs 960 GB/s GDDR6 with a **6 MB L2 *and* a
>   96 MB Infinity Cache (L3)**. That large cache hierarchy absorbs small, irregular, or
>   repeated memory accesses, so RDNA3 is fairly forgiving of small batch sizes.
>
> Vega has **no such safety net** — 4 MB of L2 and nothing below it, so almost every
> weight fetch goes all the way to HBM2. Performance therefore depends entirely on
> keeping that 2048-bit bus **saturated**:
>
> - **Too small a micro-batch (`ub=64`)** doesn't put enough independent work in flight
>   to keep the memory controller's request queues full. The bus idles between fetches,
>   HBM2 latency is exposed, and throughput drops — the card is *bandwidth-capable but
>   under-fed*.
> - **A larger micro-batch (`ub=256`)** issues many more concurrent memory requests, so
>   the controller stays busy and the latency of each HBM2 fetch is hidden behind the
>   throughput of the others. The bus runs closer to its 484 GB/s ceiling.
>
> This is why Vega **relies on raw bandwidth rather than cache latency**: it can't cache
> its way out of a small batch the way RDNA3 can, so you feed it big micro-batches
> instead. The effect plateaus once the bus is saturated — hence 256 ≈ 512 in the table
> above, while 128 and below leave measurable throughput on the floor. On a
> cache-rich RDNA3 card the same sweep would be much flatter.

> An informal `curl` test once suggested `ub=64` was faster; `llama-bench` did **not** confirm it — `ub=128` (64's neighbor) is measurably *below* the 256 peak. That earlier result was measurement noise.

Generation (`tg128`) is flat at **~66 t/s** regardless of `-ub` — micro-batch only affects prefill, as expected.

The original `-b 16 -ub 16` was a **crash workaround** for early RADV instability — once stable, it left ~2× prefill on the table.

VRAM impact of batch size is negligible — at `-c 131072` + `-b 256` + a 20K prompt, peak VRAM was **5258 MB (65%)** with GPU pinned and GTT flat (no spill).

> If you hit instability (crashes/hangs) on your driver stack, step back toward `-b 64` or `-b 16`. On this setup (Mesa RADV, CachyOS), `-b 256` was stable across repeated runs.

### Flash Attention on gfx900 — testing the folklore

A widely-repeated claim holds that **Flash Attention is slower on Vega 64 / gfx900** ("the compute units struggle with FA's memory access patterns"). **We tested it with `llama-bench` — and on this stack it's the opposite.**

| Test | FA **on** t/s | FA **off** t/s | FA speedup |
|---|---|---|---|
| pp2048 | 773.42 | 288.24 | **2.68×** |
| pp16384 | 643.92 | 186.22 | **3.46×** |
| pp32768 | 580.47 | 133.10 | **4.36×** |
| tg128 (generation) | 68.20 | 33.85 | **2.01×** |

*(Measured `-b 256 -ub 256`, `q8_0` K-cache; the FA-off leg used f16 V-cache. The gap is far too large for the V-cache format to matter.)*

Two takeaways:
1. **Keep `-fa on`.** It is one of the most impactful flags in the config — 2.7–4.4× prefill and ~2× generation. FA-off would be crippling at 128K.
2. **The claim is likely stale/misattributed.** llama.cpp's Vulkan FA shaders have improved, and current Mesa/RADV handles them well on gfx900. Also, **FA is what makes quantized (`q8_0`) KV efficient** — another reason to keep it on.

> The lesson: **measure hardware folklore on your own card.** A claim that's true for one driver/model/context can be flatly wrong for yours — the only way to know is to benchmark it.

### Power & voltage tuning — where prefill actually tops out

Prefill on this card is **compute-bound**, so it's natural to ask whether raising the
GPU power limit (via LACT or `pp_od_clk_voltage`) buys more throughput. It does — but
only up to a point, and that point is **not** the card's rated power. Everything below
was measured with `llama-bench` (`-b 256 -ub 256 -fa on`, `q8_0` KV, `-r 5`), clocks
pinned, on llama.cpp build 10107.

#### Power-target sweep (Sapphire NITRO+ RX Vega 64, 345 W TDP)

| Power target | Actual draw | pp2048 | pp16384 | pp32768 | tg128 |
|---|---|---|---|---|---|
| 200 W | ~200 W (clamped) | 681.05 | 563.14 | 482.40 | 65.59 |
| **250 W** | **~250 W** | **839.94** | **722.67** | **624.70** | **65.40** |
| 300 W (stock V) | ~250 W* | 847.98 | 724.03 | 626.71 | 66.11 |
| 360 W (stock V) | ~250 W* | 853.81 | 729.83 | 630.86 | 66.29 |

\*The card **cannot reach** a 300–360 W draw at these clocks — see below.

**The knee is ~250 W, and it's not the power limit.** Going 200 → 250 W is a real
**+23–30% prefill** gain (200 W was genuinely clamping clocks). But 250 → 300 → 360 W
does **nothing** (853 vs 847 vs 840 is noise). The reason surfaced while watching clocks
live: under sustained prefill load the core **clock-stretches back to ~1536 MHz**
(voltage modulating ~1031–1137 mV) and **never holds the 1630 MHz boost state**, no
matter how much power is *permitted*. So the real ceiling is a **thermal / clock-stretch
limit around ~1536 MHz**, not watts. Above ~250 W of actual draw there's simply no more
work the card will do — a higher power target just goes unused.

> **More watts don't help; better cooling (or a good undervolt) does.** The only way to
> lift prefill further would be sustaining a higher core clock — a cooling problem, not a
> power-budget one.

#### Undervolt is essentially free

The card was run with an aggressive undervolt (**~1602 MHz @ 990 mV** top state, vs the
stock **1630 MHz @ 1200 mV** — a ~200 mV reduction). Comparing it against stock voltage
at matched settings:

| Config | Voltage (top) | pp2048 | pp16384 | pp32768 | tg128 | Draw |
|---|---|---|---|---|---|---|
| **Undervolt** | 990 mV | 839.94 | 722.67 | 624.70 | 65.40 | ~250 W |
| Stock @ 360 W | 1200 mV | 853.81 | 729.83 | 630.86 | 66.29 | >250 W |

The undervolt costs about **+1% prefill and 0% generation** while drawing ~250 W and
running much cooler/quieter. Because lower voltage means less heat, the undervolt can
actually *sustain* clocks as well as thermally-limited stock — which is why the two
nearly tie despite the 210 mV difference. **For a 24/7 inference box, the aggressive
undervolt is the recommended operating point.**

Reset to stock (to compare, or if a profile misbehaves) and restore an undervolt via
LACT, or at the sysfs level:
```bash
echo "r" | sudo tee /sys/class/drm/card1/device/pp_od_clk_voltage   # restore firmware defaults
echo "c" | sudo tee /sys/class/drm/card1/device/pp_od_clk_voltage   # commit
```
Prefer LACT for a full multi-state undervolt curve so its daemon re-applies the same
profile on boot (otherwise you may be silently running an old profile — as happened
here, which is why early "250 W ceiling" runs were really the *undervolt* ceiling).

#### Generation is a hard bandwidth wall

Across **every** configuration tested — 200/250/300/360 W, undervolt and stock —
generation stayed pinned at **~65–66 t/s**. The memory clock (MCLK state 3, **945 MHz**)
never changed, and generation is HBM2-**bandwidth-bound** (see the Vega-vs-RDNA3 sidebar
above). No amount of core power or voltage moves it; only a memory overclock could — and
that's **not** advised here, since HBM2 errors tend to be *silent* (wrong tokens, no
crash) on an inference card.

> **A version footnote:** updating llama.cpp from build 9840 → 10107 (~250 upstream
> builds) was **perf-neutral** for this workload (prefill within ±0.7% at matched power).
> The committed numbers hold across that range — the standalone build was already at the
> gfx900 Vulkan plateau.

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
2. **Silent spill into system RAM (GTT)** — the RADV/Mesa driver satisfies the over-allocation by mapping host RAM (GTT) instead of failing. The server runs, but part of the working set is in RAM, and throughput drops as data crosses PCIe.

### The `radeontop` method

Stream stats to stdout once per second:

```bash
radeontop -d - > vram_trace.log
```

`radeontop` reports **two** memory figures — watch both:

- **`vram`** — dedicated GPU memory.
- **`gtt`** — system RAM mapped for the GPU. **A rising `gtt` under load is the spill.**

(Kernel-level truth, if you prefer: `cat /sys/class/drm/card1/device/mem_info_vram_used` and `.../mem_info_gtt_used`.)

| Signature | `vram` | `gtt` | GPU % | Meaning |
|---|---|---|---|---|
| **Healthy (resident)** | steady, < ~95% | flat at baseline (~1 GB here) | high (~94%) | Fully in VRAM ✅ |
| **Silent spill** | pinned ~97–99% | jumps well above baseline | high, but partly PCIe-wait | Overflowed into RAM ⚠️ |
| **CPU bottleneck** | moderate | flat | **low (~25%)** | GPU starved by CPU layers |

> **GPU % alone is misleading.** A spilling run can still show ~98% GPU because the card is busy *waiting on* data crossing PCIe. The figure that actually exposes a spill is **GTT**; the figure that exposes a CPU bottleneck is a **low GPU %**.

For reference, healthy E4B runs (above) hold GTT flat at ~970–1110 MB across 32K→128K. Anything markedly higher under load is spill.

## Can this card run a 12B?

Short answer: **yes — ~23 tok/s at 8K context, text-only.** Tested with `unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL` (~6.85 GB weights, **40 dense layers, no GQA**).

The core squeeze: weights alone eat ~6.85 GB of ~7.4 GB usable VRAM, leaving almost nothing for KV cache + buffers. You can keep all layers on GPU **or** keep everything in VRAM — not both:

| Config | VRAM | GTT | GPU % | tok/s (served, 8K) |
|---|---|---|---|---|
| `-ngl 99`, q8 KV, mmproj, 16K | 7877 MB (97%) | ~2496 MB | 98% | — |
| `-ngl 99`, q8 KV, no mmproj, 16K | — | — | ~96% | **21.72** |
| `-ngl 99`, q4 KV, no mmproj, 8K | 8036 MB (99%) | ~1505 MB | 96% | **22.71** |
| `-ngl 36`, q4 KV, no mmproj, 8K | 6206 MB (76%) | ~1134 MB | **~25%** | **8.64** |

### Counterintuitive result: let it spill, don't offload to CPU

The standout finding: **the GTT-spilling `-ngl 99` runs (~22 tok/s) are ~2.6× faster than the fully-resident `-ngl 36` run (8.6 tok/s)** — the opposite of what you'd assume.

- **`-ngl 36`** moves 4 layers' *compute* onto the **Ryzen 7 2700X** (Zen+, 2018). The GPU stalls each token waiting on the slow CPU → GPU drops to ~25% → **8.6 tok/s**.
- **`-ngl 99`** keeps all 40 layers *computing on the GPU*; only some weights/buffers live in host RAM and stream over PCIe. GPU stays ~96% busy → **~22 tok/s**.

> **On a weak-CPU / capable-GPU box, the driver's GTT spill beats CPU offload.** PCIe bandwidth feeding the GPU is far faster than making an old CPU do the matrix math. CPU offload (`-ngl <max`) is the worse option here.

Also worth noting: **q4 vs q8 KV barely affects speed** (22.71 vs 21.72) — the KV quant buys you *more context*, not throughput.

### `llama-bench` cross-check — and why context matters

The served figures above were re-validated with **`llama-bench`** (build 10107, undervolt
@ 250 W, direct-file load). The key lesson: **generation speed depends on how full the KV
cache is**, because the 12B is bandwidth-bound — so you must benchmark *at depth*
(`-d <ctx>`), not from an empty cache:

| `llama-bench` run | KV depth at gen | tg128 | Note |
|---|---|---|---|
| `-p 2048 -n 128` | ~empty | **~29.5** | misleadingly high — tiny cache |
| `-p 8192 -n 128` | ~empty at gen start | **29.55 ± 0.03** | prefill fills, but tg starts near-empty |
| **`-n 128 -d 8192`** | **filled to 8K** | **22.93 ± 0.28** | ✅ matches served |

**`llama-bench -d 8192` gives 22.93 tok/s — within ~1% of the served 22.71.** Two
independent methods (bench and `llama-server` + opencode) agree, so **~23 tok/s at 8K is
the trustworthy 12B number.** The ~29.5 from shallow-cache runs is an artifact of an empty
KV cache and does **not** reflect real use — it does **not** mean the 12B got faster.

Watch generation fall as the cache fills — a textbook bandwidth-wall demonstration:
**~29.5 tok/s near-empty → 22.93 at 8K (−22%)**. More resident KV = more bytes streamed
per token = lower generation. GTT stayed flat (~1110–1126 MB) across all these runs, so
the slowdown is the **KV-cache bandwidth cost itself**, not extra PCIe spill.

> **Always benchmark generation at realistic depth.** A bare `llama-bench` generation test
> runs from a near-empty cache and *overstates* tok/s for a bandwidth-bound model. Use
> `-d <ctx>` to pre-fill the cache to the context you actually run at.
>
> **Model-name note:** llama.cpp labels this GGUF `Q4_K - Medium`, but that's its generic
> quant-type guess — it misreports Unsloth Dynamic quants. The filename
> (`gemma-4-12b-it-UD-Q4_K_XL.gguf`) is authoritative; it *is* the `UD-Q4_K_XL` build.

**Verdict:** the 12B is viable at **~23 tok/s** (8K context) if you drop multimodal (`--no-mmproj`) and accept ~8–16K context and zero VRAM headroom. But for coding via opencode — where the 4B's **128K context** and headroom matter — E4B QAT remains the better pick.

## Multimodal note

E4B is multimodal — the server loads an `mmproj-BF16.gguf` projector automatically, enabling **image** (and experimental **audio**) input. It adds ~0.5 GB VRAM.

**Should you disable it with `--no-mmproj` for coding?** On E4B, **no — keep it.** The projector only runs when you actually send an image; for pure text/coding it sits idle and costs **no tok/s**, only the ~0.5 GB VRAM.

Next: [04 — opencode integration](04-opencode.md)
