# tinyinference

A minimal Rust TUI for launching and managing `llama-server` with GGUF models.

It is designed to make large, capable LLMs runnable on low-spec, low-RAM
machines without a GPU, using CPU inference and file-backed model weights. It
will not be fast.

## Requirements

- [Rust](https://www.rust-lang.org/tools/install)
- [`llama-server`](https://github.com/ggml-org/llama.cpp) from llama.cpp
- A GGUF model available locally or on Hugging Face

`llama-server` must be on `PATH`, or you can set its full executable path from
tinyinference's Configure screen.

## Run

Clone the repository, then start the development build:

```powershell
git clone https://github.com/jacobzymet/tinyinference.git
cd tinyinference
cargo run
```

On first launch, tinyinference checks for `llama-server`. If it cannot find it,
press `Enter` or `c` to open the executable-path setting.

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

Press `c` to configure the model, `llama-server` path, host, port, and runtime
settings. Press `Enter` on a value to type an exact value; use `s` to save.

Switch **Model source**, then edit the next row to enter either a Hugging Face
`owner/model` repository or a full local `.gguf` path. A local model's size is
detected automatically.

Settings are saved in the platform configuration directory. To use a portable
profile, copy the example and pass it explicitly:

```powershell
Copy-Item tinyinference.example.toml tinyinference.toml
cargo run -- --config .\tinyinference.toml
```

`tinyinference.toml` is ignored by Git, so local paths and preferences remain
local. Advanced llama.cpp options can be added through `server.extra_args`.

Useful commands:

```powershell
cargo run -- --start
cargo run -- --print-command
```

## Controls

| Key | Action |
| --- | --- |
| `s` | Start or stop the server; save from Configure |
| `r` | Restart the server |
| `c` | Configure |
| `l` | View logs |
| `?` | Help |
| `q` | Quit and stop the managed server |
| `Up` / `Down` or `k` / `j` | Select a setting |
| `Left` / `Right` or `h` / `l` | Adjust a setting |
| `Enter` | Type an exact value or toggle |

## How low-RAM operation works

With `mmap`, model weights are read-only file-backed pages, so the GGUF file
does not need to fit entirely in resident RAM. RAM is still needed for the KV
cache, compute buffers, and server state; with little RAM, storage I/O can make
inference extremely slow. `gpt-oss-120b` has 117B total parameters but
activates 5.1B per token.

The server binds to `127.0.0.1` by default and has no authentication. Configure
authentication and firewalling before exposing it to a network.

## References

- [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [gpt-oss architecture](https://openai.com/index/introducing-gpt-oss/)
