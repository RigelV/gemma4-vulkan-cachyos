# 02 — Vulkan-accelerated llama.cpp in Unsloth Studio

> **This page is only relevant if you have Unsloth Studio installed and want *its* bundled server to be GPU-accelerated.**
>
> You do **not** need any of this to run the model. The `llama-server` you built in [step 01](01-build-vulkan.md) runs directly from its build directory — that's exactly what [`run_server.sh`](../run_server.sh) does.

As of **Unsloth v0.1.50-beta** ([PR #5819](https://github.com/unslothai/unsloth/pull/5819), merged 2026-07-09), Studio has **official Vulkan support** — it will download and use the upstream ggml-org Vulkan `llama.cpp` prebuilt on AMD/Intel hosts. This supersedes the manual binary swap that used to be required. Use the official path (below); the manual swap is kept at the end only as a fallback for older Unsloth versions or fully custom builds.

## Official Vulkan support (Unsloth ≥ v0.1.50-beta)

### The one thing to get right: `UNSLOTH_FORCE_VULKAN` is an **install-time** flag

The flag opts your host into the Vulkan prebuilt **when the installer runs** — it is **not** read when the server launches. This is the single most common mistake:

```bash
# ❌ Does NOTHING — the install already happened; nothing reads the flag at launch
UNSLOTH_FORCE_VULKAN=1 unsloth studio -H 192.168.88.240 -p 8888

# ✅ Export it BEFORE the installer/setup runs, so it pulls the Vulkan build
export UNSLOTH_FORCE_VULKAN=1
# ...then trigger the llama.cpp install/setup (a fresh Studio start that installs
#    llama.cpp, or the installer entrypoint)...
```

Why it works this way: the installer's `force_vulkan_requested()` reads `UNSLOTH_FORCE_VULKAN` and, for a Vulkan-capable host, downloads `llama-<tag>-bin-ubuntu-vulkan-x64.tar.gz` from **upstream ggml-org** (the Unsloth fork manifest ships no Vulkan asset). At **runtime**, Studio decides "is this a Vulkan build?" purely by looking at the **installed files** — specifically whether `libggml-vulkan.so` sits next to `llama-server` (and no `libggml-cuda.so` / `libggml-hip.so` alongside it). So the flag's whole job is to get the right library onto disk; once it's there, Studio auto-detects Vulkan with no env var on the launch line.

> **Why AMD needs the flag.** By default Studio installs the **ROCm** build for an AMD GPU. On gfx900 (Vega) that ROCm build may not even work (ROCm dropped gfx900 after 5.6). `UNSLOTH_FORCE_VULKAN=1` at install time routes you to the upstream Vulkan prebuilt instead. (The flag only affects the llama.cpp inference backend; the torch/training stack installs separately and still sees the real GPU.)

### Steps

1. **Remove the current (ROCm/CPU) llama.cpp install** so a fresh one is pulled. Find it first:
   ```bash
   find ~ -name llama-server -path '*llama.cpp*' 2>/dev/null
   # e.g. ~/.unsloth/llama.cpp/build/bin/llama-server
   ```
2. **Re-install with the flag exported:**
   ```bash
   export UNSLOTH_FORCE_VULKAN=1
   unsloth studio -H 192.168.88.240 -p 8888   # triggers a fresh install -> pulls the Vulkan build
   ```
3. **Verify Vulkan is active** — two independent checks:
   - The library is present (this is what runtime detection keys off):
     ```bash
     ls <install_dir>/build/bin/ | grep -E 'ggml-(vulkan|cuda|hip)'
     # WANT: libggml-vulkan.so   and NO libggml-cuda.so / libggml-hip.so
     ```
   - Studio's log shows the Vulkan probe reading VRAM at startup:
     ```
     {"level":"info","event":"Vulkan GPU memory detected: VK0=7029MiB"}
     ```
     `VK0=…MiB` is llama.cpp's own Vulkan library reporting free VRAM — proof the backend engaged.

### Dry-run the selection (optional)

To confirm the flag routes correctly *without* installing, use the resolver — it should report the `linux-vulkan` upstream asset when the flag is set:

```bash
UNSLOTH_FORCE_VULKAN=1 python <path>/install_llama_prebuilt.py --resolve-prebuilt <tag> --simple-policy
```

### Notes

- **Studio pulls the current upstream release.** In this session Studio installed `b10092` — the very latest at the time — so its Vulkan build is not stale.
- **Updates preserve Vulkan.** The installer re-asserts `UNSLOTH_FORCE_VULKAN=1` automatically when the installed asset name contains `vulkan`, so an update won't silently re-route you back to ROCm/CUDA.
- **Intel GPUs** get this automatically (no flag needed) — Studio detects Intel and picks Vulkan. The flag is the AMD opt-in.

---

## Legacy / fallback: manual binary swap

> **Only needed on Unsloth versions older than v0.1.50-beta, or for a fully custom build.** On current Unsloth, use the official path above instead.

Older Studio shipped its **own** pre-built `llama.cpp` (a CPU build) with no way to request Vulkan. The steps below replace that bundled binary with the Vulkan build you compiled in step 01 so Studio's server uses the GPU.

### 1. Find the real binary (not a log file)

A naive `find ~/.unsloth -name llama-server` can match a **log file** at `~/.unsloth/studio/logs/llama-server`. Filter to executables:

```bash
find ~/.unsloth -type f -executable \( -name "llama-cli" -o -name "llama-server" \) 2>/dev/null
```

On this setup that returns:

```
/home/<user>/.unsloth/llama.cpp/build/bin/llama-server
```

Studio bundles only `llama-server` (no `llama-cli`).

### 2. Understand the symlink layout

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

### 3. Swap

```bash
DST=~/.unsloth/llama.cpp/build/bin
cp "$DST/llama-server" "$DST/llama-server.cpu.bak"          # backup the CPU build
cp ~/llama.cpp/build/bin/llama-server "$DST/llama-server"   # drop in the Vulkan build

# verify it offloads:
"$DST/llama-server" -m <some-model>.gguf -ngl 99 -b 16      # expect Vulkan0 ~4GB
```

> **Note:** a manual swap of just `llama-server` may not bring the sibling `libggml-vulkan.so` the binary needs. The official install path above handles the whole payload (server + ggml backend libs) correctly, which is another reason to prefer it on current Unsloth.

### 4. Caveat: updates overwrite this

Studio's path is `~/.unsloth/llama.cpp/build/...`, which means an update (`curl -fsSL https://unsloth.ai/install.sh | sh`) is likely to **rebuild/replace** the binary. **Re-run the copy after any update** — or, better, move to the official `UNSLOTH_FORCE_VULKAN` path, which survives updates.

Next: [03 — Run parameters and tuning](03-run-tuning.md)
