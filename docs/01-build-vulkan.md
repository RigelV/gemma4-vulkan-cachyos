# 01 — Building llama.cpp with Vulkan on CachyOS

The goal: a `llama-server` / `llama-cli` that offloads to the Vega 64 via the **RADV** Vulkan driver, since ROCm no longer supports gfx900.

## 1. Confirm RADV / Vulkan is present

```bash
vulkaninfo | grep -i "radv\|deviceName"
```

You should see your Radeon device listed via the RADV driver. If `vulkaninfo` is missing, install `vulkan-tools` (below).

## 2. Install CachyOS dependencies

The build needs the Vulkan **headers** first, then **SPIRV-Tools** for shader compilation. Installing in this order avoids the missing-header errors that otherwise stop the CMake configure step.

```bash
sudo pacman -S --needed \
  base-devel cmake git \
  vulkan-headers vulkan-tools \
  vulkan-radeon \
  spirv-tools spirv-headers \
  shaderc glslang
```

| Package | Why |
|---|---|
| `vulkan-headers` | Build-time Vulkan API headers (needed before configure) |
| `vulkan-radeon` | RADV (Mesa) Vulkan driver for AMD |
| `vulkan-tools` | `vulkaninfo` for verification |
| `spirv-tools` / `spirv-headers` | Compile GLSL compute shaders to SPIR-V |
| `shaderc` / `glslang` | Shader toolchain used by the Vulkan backend |

## 3. Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

The key flag is **`-DGGML_VULKAN=ON`**. Binaries land in `build/bin/`.

## 4. Verify GPU offload

Run any GGUF with full offload (`-ngl 99`) and watch the load log:

```
load_tensors:      Vulkan0 model buffer size =  4095.05 MiB
load_tensors:   CPU_Mapped model buffer size =    70.31 MiB
```

- **`Vulkan0 ... = ~4 GB`** → weights are on the GPU.
- The small **`CPU_Mapped`** amount is just the embeddings / layer norms that aren't offloaded — normal.

### Harmless debug line

With verbose output you may see:

```
D done_getting_tensors: tensor 'token_embd.weight' (q4_K) (and 0 others) cannot be used with preferred buffer type Vulkan_Host, using CPU instead
```

This is **expected**. The token embedding table is intentionally kept in pinned CPU memory (`Vulkan_Host`); it's the same ~70 MiB shown above. "(and 0 others)" confirms only embeddings fell back. Not an error.

> You'd only worry if **dozens** of tensors fell back to CPU, or the `Vulkan0` buffer dropped near 0 — that would mean offload broke.

Next: [02 — Swapping the binary into Unsloth Studio](02-studio-swap.md)
