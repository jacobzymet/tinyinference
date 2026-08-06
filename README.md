# <img src="assets/ti.png" alt="" width="42" align="top"> tinyinference

A minimal, lightweight Rust desktop app for launching and managing `llama-server` with GGUF models.

> **Note:** tinyinference started as a terminal UI (TUI), then became a local web UI. The control panel now opens in its own native window. The web server is still there underneath — the chat page runs in your browser, and `--no-window` gives you the old browser-only behaviour.

A primary feature/appeal of tinyinference is the seamless ability to make large, capable LLMs runnable on low-spec, low-RAM machines without a GPU, using CPU inference and file-backed model weights. **It will not be fast, in fact, it will often be painfully slow. The point is that it runs at all on low-spec hardware, which is pretty cool.**

**This feature uses mmap**, and [you can read more about how it works and its integration into tinyinference here.](https://jacobzymet.com/notes/running-oversized-gguf-language-models-on-low-ram-hardware-using-system-memory-mapping)

## Requirements

- [Rust](https://www.rust-lang.org/tools/install)
- [`llama-server`](https://github.com/ggml-org/llama.cpp) from llama.cpp
- A GGUF model available locally or on Hugging Face
- A modern web browser (for the chat page)
- A system webview for the control-panel window:
  - **Windows** — WebView2, preinstalled on Windows 11
  - **macOS** — WKWebView, part of the OS
  - **Linux** — WebKitGTK development packages, e.g.
    `libwebkit2gtk-4.1-dev` and `libxdo-dev` on Debian/Ubuntu. To skip this
    entirely, build with `--no-default-features` (see [Windowless](#windowless)).

`llama-server` must be on `PATH`, or you can set its full executable path from
tinyinference's Configure tab.

## Run

Clone the repository, then start the development build:

```powershell
git clone https://github.com/jacobzymet/tinyinference.git
cd tinyinference
cargo run
```

The control panel opens in its own window. On first launch, tinyinference checks
for `llama-server`. If it cannot find it, use the prompt to open the
executable-path setting.

Only one instance runs per address. Launching tinyinference again raises the
window that is already open rather than starting a second server — so a desktop
shortcut behaves the way you would expect. If the address is held by an
unrelated program, tinyinference says so and exits instead of guessing; use
`--bind` to pick another. Two instances *can* coexist on different addresses.

### Two surfaces

The **control panel** is the native window: models, runtime settings, logs, and
live statistics. **Chat** is a separate page served at `/chat` that opens in your
default browser, so conversations live alongside your normal tabs. The window
hosts only the control panel — the Chat button, the llama-server UI link, and
any other outbound link are handed to your browser rather than opened in-window.

### Windowless

To run headless, or on a machine without a system webview, skip the window:

```powershell
cargo run -- --no-window
```

Then open the printed URL yourself (default `http://127.0.0.1:3920`), or use
`--open` to launch the browser automatically (this implies `--no-window`):

```powershell
cargo run -- --open
```

To drop the windowing dependencies at build time entirely:

```powershell
cargo build --release --no-default-features
```

## Build

Build an optimized executable:

```powershell
cargo build --release
```

Run it on Windows:

```powershell
.\target\release\tinyinference.exe
```

On macOS or Linux, use:

```sh
./target/release/tinyinference
```

## Configuration

Open the **Models** tab to manage your model library. Open **Configure** for the
`llama-server` path, host, port, runtime preset, and other server settings. Use
**Save** on Configure to write runtime settings to disk; library changes
(add / use / remove / import) save automatically.

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

The server listen address (used by the window, the chat page, and the API) can
be set before the UI is up, in priority order:

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
cargo run -- --no-window
cargo run -- --open
cargo run -- --bind 127.0.0.1:4000
cargo run -- --print-command
```

## UI

| Action | Where |
| --- | --- |
| Start / stop | Header button |
| Restart | Header button |
| Chat with the model | Header **Chat** button (opens in your browser) |
| Manage models | Models tab |
| Configure runtime / server | Configure tab |
| View logs | Logs tab |
| Live statistics | Stats tab |
| Copy OpenAI-compatible `/v1` URL | Dashboard |
| Copy resolved `llama-server` command | Dashboard |
| Save settings | Models, Configure, or Dashboard |

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

The tinyinference server binds to `127.0.0.1:3920` by default (override with
`--bind`, `TINYINFERENCE_BIND`, or `[ui]` in the config). It has no
authentication: anything that can reach it can drive the model, read the logs,
and change the configuration — running it in a window does not hide it from the
network. The managed `llama-server` binds to `127.0.0.1:8080` by default and is
likewise unauthenticated. Configure authentication and firewalling before
exposing either to a network.

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

## References

- [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [gpt-oss architecture](https://openai.com/index/introducing-gpt-oss/)
