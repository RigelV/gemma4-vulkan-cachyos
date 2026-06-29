# 02 — (Optional) Swapping the Vulkan binary into Unsloth Studio

> **This page is only relevant if you have Unsloth Studio installed and want *its* bundled server to be GPU-accelerated.**
>
> You do **not** need any of this to run the model. The `llama-server` you built in [step 01](01-build-vulkan.md) runs directly from its build directory — that's exactly what [`run_server.sh`](../run_server.sh) and the [opencode guide](04-opencode.md) use. If you don't use Studio, **skip to [03 — Run parameters and tuning](03-run-tuning.md).**

Unsloth Studio ships its **own** pre-built `llama.cpp` (a CPU build). The steps below replace that bundled binary with the Vulkan build so Studio's own server uses the GPU.

## 1. Find the real binary (not a log file)

A naive `find ~/.unsloth -name llama-server` can match a **log file** at `~/.unsloth/studio/logs/llama-server`. Filter to executables:

```bash
find ~/.unsloth -type f -executable \( -name "llama-cli" -o -name "llama-server" \) 2>/dev/null
```

On this setup that returns:

```
/home/<user>/.unsloth/llama.cpp/build/bin/llama-server
```

Studio bundles only `llama-server` (no `llama-cli`).

## 2. Understand the symlink layout

There is a convenience symlink pointing **down** into the build dir:

```
~/.unsloth/llama.cpp/llama-server -> build/bin/llama-server
```

So `build/bin/llama-server` is the **real** file; the top-level path is just a link to it. Copying onto the real file means the link follows automatically — no symlink trap.

Confirm with:

```bash
readlink -f ~/.unsloth/llama.cpp/build/bin/llama-server
# -> /home/<user>/.unsloth/llama.cpp/build/bin/llama-server  (the real file)
```

## 3. Swap

```bash
DST=~/.unsloth/llama.cpp/build/bin
cp "$DST/llama-server" "$DST/llama-server.cpu.bak"          # backup the CPU build
cp ~/llama.cpp/build/bin/llama-server "$DST/llama-server"   # drop in the Vulkan build

# verify it offloads:
"$DST/llama-server" -m <some-model>.gguf -ngl 99 -b 16      # expect Vulkan0 ~4GB
```

## 4. Caveat: updates overwrite this

Studio's path is `~/.unsloth/llama.cpp/build/...`, which means an update (`curl -fsSL https://unsloth.ai/install.sh | sh`) is likely to **rebuild/replace** the binary. **Re-run the copy after any Studio upgrade.**

Next: [03 — Run parameters and tuning](03-run-tuning.md)
