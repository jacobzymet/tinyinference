# <img src="assets/ti.png" alt="" width="42" align="top"> tinyinference

A minimal, lightweight Rust web app for launching and managing `llama-server` with GGUF models.

> **Note:** tinyinference started as a terminal UI (TUI), then a local web UI with a native window. It is now browser-only: one local server, with **Chat** at `/` and **Admin** at `/admin`.

A primary feature/appeal of tinyinference is the seamless ability to make large, capable LLMs runnable on low-spec, low-RAM machines without a GPU, using CPU inference and file-backed model weights. **It will not be fast, in fact, it will often be painfully slow. The point is that it runs at all on low-spec hardware, which is pretty cool.**

**This feature uses mmap**, and [you can read more about how it works and its integration into tinyinference here.](https://jacobzymet.com/notes/running-oversized-gguf-language-models-on-low-ram-hardware-using-system-memory-mapping)

An additional benefit of tinyinference is that all you need to get going is a single binary. You don't need to "install" tinyinference. You download one portable binary and replace it as updates come. Of course, you still need llama.cpp, but that's besides the point.

## Requirements

- [Rust](https://www.rust-lang.org/tools/install) (from source)
- [`llama-server`](https://github.com/ggml-org/llama.cpp) from llama.cpp
- A GGUF model available locally or on Hugging Face
- A modern web browser

`llama-server` must be on `PATH`, or you can set its full executable path from
Admin → Config.

## Run

Download a release binary from [GitHub Releases](https://github.com/jacobzymet/tinyinference/releases) and run it directly.

On Windows:

```powershell
.\tinyinference.exe
```

On macOS or Linux:

```sh
./tinyinference
```

Then open the printed URL (default `http://127.0.0.1:3920`), or pass `--open` to
launch the browser automatically.

### From source

```powershell
git clone https://github.com/jacobzymet/tinyinference.git
cd tinyinference
cargo run -- --open
```

On first launch, tinyinference checks for `llama-server`. If it cannot find it,
use the prompt in Admin to set the executable path.

Only one instance runs per address. Launching again reports that the server is
already running instead of starting a second one. If the address is held by an
unrelated program, tinyinference says so and exits; use `--bind` to pick another.
Two instances *can* coexist on different addresses.

### Chat and Admin

Both surfaces share the same local server and a Chat | Admin mode switch:

| Path | Surface |
| --- | --- |
| `/` | **Chat** — conversations, projects, agent mode |
| `/admin` | **Admin** — models, runtime, devices, logs, stats |

`/chat` permanently redirects to `/`.

## Build

The release binary is **self-contained**: chat HTML, admin HTML, `orb.js`, and
icons are compiled into the executable (`include_str!` / `include_bytes!` in
`src/web/embed.rs`). You ship one file — nothing else from this repo needs to sit
beside it (you still need `llama-server` on the machine).

### This machine

```powershell
# Windows
.\scripts\build-release.ps1
.\dist\tinyinference-windows-x86_64.exe
```

```sh
# macOS / Linux
./scripts/build-release.sh
./dist/tinyinference-macos-aarch64   # or linux-x86_64, etc.
```

Or plain Cargo:

```powershell
cargo build --release
```

### Windows + macOS + Linux

Cross-building every OS from one laptop is unreliable (macOS especially needs a
Mac). The **Release** GitHub Action builds all of them:

| Artifact | Notes |
| --- | --- |
| `tinyinference-windows-x86_64.exe` | Windows |
| `tinyinference-macos-aarch64` | Apple Silicon |
| `tinyinference-macos-x86_64` | Intel Mac |
| `tinyinference-linux-x86_64` | Linux |

```sh
# Manual run (uploads artifacts; tagging also publishes a GitHub Release)
gh workflow run release.yml

# Or push a version tag
git tag v0.3.1
git push origin v0.3.1
```

## Configuration

Open **Admin → Models** to manage your model library. Open **Admin → Config** for
the `llama-server` path, port, runtime preset, and other server settings. The
listen address is owned by **Devices** (loopback when Share is off). Changes
autosave immediately — no manual save step.

On **Models**:
- **My models** is your explicit library. **Add model** registers a Hugging Face
  `owner/model` (URL paste works) or a local `.gguf` path. Hugging Face models
  can download into the local hub cache with live progress.
- **Use**, **Download**, and **Remove** act on library entries. Remove deletes
  the entry and local cache files after confirmation, and does not bring the
  default model back.
- **Found on disk** lists autodiscovered GGUF caches that are not in your
  library yet. **Import** adds one to the library; **Delete files** removes the
  cache only.

Settings are saved in the platform configuration directory. To use a portable
profile, copy the example and pass it explicitly:

```powershell
Copy-Item tinyinference.example.toml tinyinference.toml
cargo run -- --config .\tinyinference.toml
```

`tinyinference.toml` is ignored by Git, so local paths and preferences remain
local. Advanced llama.cpp options can be added through `server.extra_args`.

The server listen address (used by chat, admin, and the API) can be set before
the UI is up, in priority order:

1. `--bind 127.0.0.1:4000`
2. environment variable `TINYINFERENCE_BIND=127.0.0.1:4000`
3. `[ui]` in the config file (`host` / `port`, default `127.0.0.1:3920`)

```toml
[ui]
host = "127.0.0.1"
port = 3920
```

Useful commands:

```powershell
cargo run -- --start
cargo run -- --open
cargo run -- --bind 127.0.0.1:4000
cargo run -- --print-command
```

## UI

| Action | Where |
| --- | --- |
| Chat with the model | `/` (Chat) |
| Start / stop / restart | `/admin` top bar |
| Manage models | Admin → Models |
| Configure runtime / server | Admin → Config |
| Share / linked devices | Admin → Devices |
| View logs | Admin → Logs |
| Live statistics | Admin → Stats |
| Copy OpenAI-compatible `/v1` URL | Admin → Dash |
| Copy resolved `llama-server` command | Admin → Dash |

When a Hugging Face model has to be fetched, the status reads `downloading`
instead of `starting`, with a progress bar, transfer rate, and time remaining.
`llama-server` prints nothing while it downloads, so progress is measured from
the cached file as it grows; its real size comes from the Hugging Face file
listing, matched to the file being written by its object id. The status changes
to `starting` once the weights begin loading. Without network access the bytes
fetched are still reported, only without a percentage.

The statistics tab shows endpoint state, PID, uptime, process CPU and resident
RAM, plus request and token counters and throughput from `llama-server`.
tinyinference enables llama.cpp's local metrics endpoint for managed servers;
metrics remain marked unavailable until the server is ready. Clipboard copy uses
the browser clipboard API, with a system clipboard fallback via `clip.exe` on
Windows, `pbcopy` on macOS, and `wl-copy`, `xclip`, or `xsel` on Linux.

tinyinference binds to `127.0.0.1:3920` by default (override with `--bind`,
`TINYINFERENCE_BIND`, or `[ui]` in the config) and stays private. Use **Admin →
Devices** to expose only the managed `llama-server` OpenAI-compatible inference
API on Tailscale only (`100.x`, default), LAN (all interfaces), or a specific
address. Chat and Admin always stay on loopback. While sharing, tinyinference
disables the llama web UI, `/slots`, and `/metrics`, enables API keys, and serves
**HTTPS with a self-signed certificate** stored under your config directory
(`tls/cert.pem` and `tls/key.pem`). Clients must trust that certificate
(browser/OS warning is expected) and send `Authorization: Bearer …`. Your
`llama-server` build needs OpenSSL support (`LLAMA_OPENSSL=ON`). Note: llama.cpp
still serves `/health` and `/models` without a key; completions and most other
routes require the API key. Prefer Tailscale. Restart the model after changing
share settings or keys. You can also point local chat at a remote
OpenAI-compatible base URL via Linked LLM.

You can run more than one model at a time: **Start another** launches the
currently configured model on the next free port. The dashboard lists running
servers; chat has a model selector at the top.

## How low-RAM operation works

With `mmap`, model weights are read-only file-backed pages, so the GGUF file
does not need to fit entirely in resident RAM. RAM is still needed for the KV
cache, compute buffers, and server state; with little RAM, storage I/O can make
inference extremely slow. `gpt-oss-120b` has 117B total parameters but
activates 5.1B per token, so it is a good default choice for tinyinference.

## Getting more performance

Most speedups come from the model and machine, not from tinyinference itself:

- Prefer smaller / more active-efficient models (MoE like gpt-oss helps; a dense
  70B on the same box will feel worse).
- Put the GGUF on the fastest local disk you have (NVMe >> HDD/network share).
  mmap latency dominates when RSS can't hold hot pages.
- Use a quant that fits your machine's working set better; shaving file size
  often beats clever flags.
- Build `llama-server` with the right CPU backend (AVX2/AVX512/ARM, ideally
  OpenBLAS/BLIS or vendor BLAS). A generic binary leaves a lot on the table.

tinyinference passes `--threads` set to the machine's physical core count when
launching `llama-server`. Override it with `server.extra_args` if needed.

By default it also enables flash attention (`--flash-attn on`) and quantized KV
cache types (`--cache-type-k/v q8_0`) so more RAM stays free for weight pages.
Turn these off or change the cache types in Configure if your `llama-server`
build rejects them.

## Credits

Chat and status “thought orb” animations are adapted from
[thinking-orbs](https://github.com/Jakubantalik/thinking-orbs) by Jakub Antalik
(MIT). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## References

- [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [gpt-oss architecture](https://openai.com/index/introducing-gpt-oss/)
