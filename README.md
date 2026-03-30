# Voxtral CLI

Local text-to-speech for Apple Silicon Macs using the quantized MLX build of `Voxtral-4B-TTS-2603`.

This project provides:

- a Rust CLI for download, serving, synthesis, and benchmarking
- a local Python MLX daemon that keeps the model warm between requests
- preset-voice speech generation with piped input or `--text`
- simple performance profiles tuned for local latency

## Status

This repo is currently focused on:

- Apple Silicon macOS
- local inference only
- preset voices only
- the quantized MLX model `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit`

It is usable today, but still a pragmatic local tool rather than a fully polished production package.

## Requirements

- macOS on Apple Silicon
- Rust toolchain
- Python 3
- `uv`

The first `download` will create a dedicated Python runtime under your local application data directory and fetch the model weights.

## Quick Start

```bash
cargo run -- download
cargo run -- serve
cargo run -- speak --text "Hello world" --profile balanced
```

Piped input also works:

```bash
echo "Hello from stdin" | cargo run -- speak
pbpaste | cargo run -- speak --profile fast
```

## Commands

### Download runtime and model

```bash
cargo run -- download
```

This bootstraps the managed Python environment and downloads the default quantized model.

### Start the daemon

```bash
cargo run -- serve
```

For debugging startup issues:

```bash
cargo run -- serve --foreground
```

### Synthesize speech

```bash
cargo run -- speak --text "Hello world"
cargo run -- speak --text "Hello world" --profile fast
cargo run -- speak --text "Hello world" --output out.wav --no-play
echo "Hello from stdin" | cargo run -- speak
```

Notes:

- `speak` plays audio by default and still saves the file
- pass `--no-play` when you only want the generated file
- if `--text` is omitted, `speak` will read from piped stdin

### List voices

```bash
cargo run -- voices
```

### Check daemon state

```bash
cargo run -- status
cargo run -- doctor
```

### Stop the daemon

```bash
cargo run -- stop
```

### Run benchmarks

```bash
cargo run -- bench --scenario short --profile balanced --runs 3
cargo run -- bench --scenario all --profile fast --runs 5
```

## Profiles

The CLI supports three synthesis profiles:

- `quality`: highest quality, slowest
- `balanced`: default profile
- `fast`: lowest latency, most aggressive tradeoff

Current local measurements on this machine during development showed roughly:

- `quality`: about `5.0x` real time on the short prompt benchmark
- `balanced`: about `6.2x` real time on the short prompt benchmark
- `fast`: about `8.1x` real time on the short prompt benchmark

Treat these as directional, not guaranteed. Results will vary with macOS version, MLX/Metal behavior, and machine configuration.

## Output and Runtime Behavior

- Audio is normalized before writing so default output is clearly audible.
- The daemon keeps the model loaded to reduce warm-request latency.
- Benchmark results are written to the local benchmark JSON path managed by the app config.
- Runtime/config/cache files live under your user application directories, not in the repo.

## Testing

Fast automated checks:

```bash
cargo test
cargo test --test cli_smoke
```

Manual daemon and synthesis checks:

```bash
cargo run -- serve --foreground
cargo run -- status
cargo run -- speak --text "hello world" --profile balanced --no-play
```

Benchmark rerun:

```bash
cargo run -- bench --scenario short --profile fast --runs 2
```

## Troubleshooting

### Daemon startup times out

Try:

```bash
cargo run -- serve --foreground
tail -n 100 "$HOME/Library/Application Support/ai.lucataco.voxtral-cli/runtime/voxtral.log"
```

If the daemon is already wedged, stop it and retry:

```bash
cargo run -- stop
cargo run -- serve
```

### No audio or very quiet audio

- `speak` plays audio by default unless `--no-play` is used
- generated files are also saved under the local outputs directory
- you can force file-only output with:

```bash
cargo run -- speak --text "hello world" --no-play --output out.wav
```

### Model/runtime issues

Reinstall the managed runtime and verify the model path:

```bash
cargo run -- download
cargo run -- doctor
```

## Security Notes

- The daemon uses a local Unix socket under your user application directory.
- Runtime directories are created with restricted permissions.
- The client includes request timeouts and basic helper-process validation before sending signals.
- This is still a local developer tool. Review the code and operational model before exposing it inside broader automation or multi-user environments.

## License and Model Usage

This repository contains local tooling. The model itself is downloaded separately from Hugging Face and remains subject to the upstream model license and usage terms.

Verify the current license and usage restrictions of:

- `mistralai/Voxtral-4B-TTS-2603`
- `mlx-community/Voxtral-4B-TTS-2603-mlx-4bit`
