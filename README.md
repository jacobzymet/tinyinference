# tinyinference

A minimal Rust web UI for launching and managing `llama-server` with GGUF models.

> **Note:** tinyinference started as a terminal UI (TUI). It now runs as a local web UI instead. Start the binary and open the printed URL in your browser (default `http://127.0.0.1:3920`).

It is designed to make large, capable LLMs runnable on low-spec, low-RAM machines without a GPU, using CPU inference and file-backed model weights. **It will not be fast, in fact, it will often be painfully slow. The point is that it runs at all on low-spec hardware, which is pretty cool.**

## Requirements

- [Rust](https://www.rust-lang.org/tools/install)
- [`llama-server`](https://github.com/ggml-org/llama.cpp) from llama.cpp
- A GGUF model available locally or on Hugging Face
- A modern web browser

`llama-server` must be on `PATH`, or you can set its full executable path from
tinyinference's Configure tab.

## Run

Clone the repository, then start the development build:

```powershell
git clone https://github.com/jacobzymet/tinyinference.git
cd tinyinference
cargo run
```

Open the printed URL in your browser (default `http://127.0.0.1:3920`), or pass
`--open` to launch it automatically:

```powershell
cargo run -- --open
```

On first launch, tinyinference checks for `llama-server`. If it cannot find it,
use the prompt to open the executable-path setting.

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

Open the **Configure** tab to edit the model, `llama-server` path, host, port,
and runtime settings. Use **Save** to write the configuration to disk.

Switch **Model source**, then edit the model field to enter either a Hugging Face
`owner/model` repository or a full local `.gguf` path. A local model's size is
detected automatically.

The **Models** dropdown lists up to eight previously used Hugging Face
repositories and local GGUF paths, and autodiscovers GGUF models already present
in the local Hugging Face hub / llama.cpp cache (including downloads performed by
`llama-server`). Recent picks are stored with the rest of the configuration when
you save.

Settings are saved in the platform configuration directory. To use a portable
profile, copy the example and pass it explicitly:

```powershell
Copy-Item tinyinference.example.toml tinyinference.toml
cargo run -- --config .\tinyinference.toml
```

`tinyinference.toml` is ignored by Git, so local paths and preferences remain
local. Advanced llama.cpp options can be added through `server.extra_args`.

The web UI listen address can be set before the UI is up, in priority order:

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
| Start / stop | Header button |
| Restart | Header button |
| Configure | Configure tab |
| View logs | Logs tab |
| Live statistics | Stats tab |
| Copy OpenAI-compatible `/v1` URL | Dashboard |
| Copy resolved `llama-server` command | Dashboard |
| Save settings | Configure tab or Dashboard |

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

The control UI binds to `127.0.0.1:3920` by default (override with `--bind`,
`TINYINFERENCE_BIND`, or `[ui]` in the config). The managed `llama-server`
binds to `127.0.0.1:8080` by default and has no authentication. Configure
authentication and firewalling before exposing either to a network.

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
