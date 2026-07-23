# tinyinference

A minimal Rust TUI for running disk-backed GGUF models with `llama-server`.
It manages one server process, exposes the important runtime settings, streams
logs, and keeps model-file size separate from resident RAM.

Designed to run large, capable LLMs on low-spec, low-RAM hardware without a
GPU by using CPU inference and disk-backed model weights. It will not be fast;
the goal is to make otherwise impractical models runnable, not responsive.

## Run

Requires Rust and `llama-server` on `PATH` (or its full path configured in the
TUI). If it is not detected at launch, the app opens a prompt that leads
directly to the executable-path setting. On Windows:

```powershell
winget install llama.cpp
cargo run
```

Useful options:

```powershell
cargo run -- --start
cargo run -- --config .\tinyinference.toml
cargo run -- --print-command
```

## Controls

Dashboard:

| Key | Action |
| --- | --- |
| `s` | Start or stop |
| `r` | Restart |
| `c` | Configure |
| `l` | Logs |
| `?` | Help |
| `q` | Quit and stop the managed server |

Configure:

| Key | Action |
| --- | --- |
| `↑` / `↓` or `k` / `j` | Select |
| `←` / `→` or `h` / `l` | Adjust |
| `Enter` | Enter an exact value or toggle |
| `Space` | Toggle |
| `s` | Save |
| `Esc` | Back |

Each selected setting includes a short explanation. Numeric settings accept
exact values; context also accepts forms such as `8k` and `16k`. Invalid input
stays open with a correction message.

Switch **Model source**, then edit the next row to enter either a Hugging Face
`owner/model` repository or a full `.gguf` path. Both values are remembered
while the program is open. Local file size is detected automatically.

For **llama-server path**, enter `llama-server` when it is on `PATH`, or paste
its full executable path. Quoted paths are accepted.

Changes made while the server is running are marked pending and take effect
after restart.

## Default profile

```text
llama-server -hf ggml-org/gpt-oss-120b-GGUF --device none
  --n-gpu-layers 0 --fit off --mmap --no-repack --no-warmup
  --ctx-size 8192 --batch-size 8 --ubatch-size 8 --parallel 1
  --cache-ram 0 --ctx-checkpoints 0 --no-mmproj --jinja
  --host 127.0.0.1 --port 8080
```

With `mmap`, model weights remain read-only file-backed pages; the GGUF file
does not need to fit in resident RAM. Low RAM can make storage I/O dominate,
while KV cache, compute buffers, and server state still require ordinary RAM.
`gpt-oss-120b` has 117B total parameters but activates 5.1B per token.

Settings are saved as TOML in the platform configuration directory after
pressing `s`. Add advanced flags to `server.extra_args`; they are appended to
the generated command.

The server binds to `127.0.0.1` by default and has no authentication. Configure
authentication and firewalling before exposing it to a network.

References: [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) and [gpt-oss architecture](https://openai.com/index/introducing-gpt-oss/).
