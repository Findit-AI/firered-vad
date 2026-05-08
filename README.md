<div align="center">
<h1>firered-vad</h1>
</div>
<div align="center">

Streaming Voice Activity Detection that wraps the [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) ONNX model. 

[<img alt="github" src="https://img.shields.io/badge/github-findit--ai/firered--vad-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Ffirered-vad" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-ai/firered-vad/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-ai/firered-vad?style=for-the-badge&token=6R3QFWRWHL&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-firered--vad-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/firered-vad?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPC9zdmc+DQo=" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/firered-vad?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge" height="22">

</div>


Streaming Voice Activity Detection that wraps the [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) ONNX model. Bit-for-bit parity with upstream Python's `FireRedStreamVad`, with a Sans-I/O Rust API designed for piping continuous human-speech windows into Whisper or any other downstream consumer.

A sibling crate to [`silero`](https://github.com/uqio/silero) for callers who want a true streaming VAD: 10 ms frame granularity, no externally-managed RNN state, and a built-in postprocessor with smoothing and a 4-state machine.

## Installation

```toml
[dependencies]
firered-vad = "0.1"
```

The default `bundled` feature embeds the ONNX model (~2.3 MB) and CMVN stats. Disable to ship your own:

```toml
[dependencies]
firered-vad = { version = "0.1", default-features = false }
```

## Examples

Please see details in [examples](./examples/).

## API at a glance

`Vad` is a single Sans-I/O state machine:

| Method | Purpose |
| --- | --- |
| `Vad::bundled()` | Construct from the bundled ONNX + CMVN with default options |
| `Vad::bundled_with(opts)` | Same, with custom `VadOptions` |
| `Vad::from_memory(model)` / `from_file(path)` | Custom model bytes/path with bundled CMVN |
| `Vad::from_memory_with_cmvn` / `Vad::from_file_with_cmvn` | Fully-custom model + CMVN |
| `Vad::from_ort_session(session, cmvn, opts)` | Wrap an externally-built `ort::Session` |
| `push_samples(&[f32])` | Feed PCM, returns the next available closed segment (or None) |
| `finish()` | Mark end-of-stream; returns the trailing segment if one was open |
| `reset()` | Wipe all per-stream state |
| `pending_segments()` | Number of buffered segments awaiting drain via `push_samples(&[])` |

## Music vs singing

The bundled FireRedVAD streaming model is trained for **voice activity** as a binary classifier: vocal sources score high regardless of whether they're speech or singing, while pure instrumental music scores low. In practice this means singing is treated as a positive segment (emitted), pure music is rejected (no segment), and speech behaves as expected. The dedicated 3-class AED model (which separates speech / singing / music explicitly) is non-streaming upstream and is not part of this crate; it would be a separate concern.

## Tuning

Options reproduce upstream `FireRedStreamVadConfig` defaults exactly. To match upstream's four "mode" presets, configure directly:

```rust
use core::time::Duration;
use firered_vad::VadOptions;

// "Permissive" preset (upstream mode 1):
let opts = VadOptions::new()
    .with_speech_threshold(0.5)
    .with_min_speech_duration(Duration::from_millis(100))
    .with_min_silence_duration(Duration::from_millis(150));

// "Aggressive" — threshold 0.7, min_speech 150 ms, min_silence 100 ms
// "Very aggressive" — threshold 0.9, min_speech 200 ms, min_silence 50 ms
// "Very permissive" — threshold 0.3, min_speech 80 ms, min_silence 200 ms
```

## Features

| Feature | Default | What it does |
| --- | --- | --- |
| `bundled` | yes | Embed the ONNX model + CMVN as `BUNDLED_MODEL` / `BUNDLED_CMVN` constants |
| `serde` | no | `Serialize` / `Deserialize` for `VadOptions` and `SessionOptions`; Duration fields use `humantime-serde` |
| `coreml`, `directml`, `cuda`, `rocm`, `tensorrt`, `openvino` | no | Pass-through to `ort` for the matching execution provider |

## Parity status

Bit-for-bit parity with upstream Python's `StreamVadPostprocessor` is the design contract. The v1 verification rests on:

- The integration test (`tests/integration_test.rs::pushing_samples_in_arbitrary_chunks_yields_identical_event_stream`) — proves the streaming pipeline is deterministic across chunk sizes.
- Hand-derived state-machine unit tests in `src/detector.rs::tests`.
- Empirical model contract verification at construction time (ONNX I/O shapes).

A per-frame numerical parity harness against the upstream Python reference (planned for `tests/parity/`) is deferred post-v1.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. The bundled FireRedVAD model and CMVN stats are Apache-2.0; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

[Github-url]: https://github.com/Findit-AI/firered-vad
[CI-url]: https://github.com/Findit-AI/firered-vad/actions/workflows/ci.yml
[codecov-url]: https://app.codecov.io/gh/Findit-AI/firered-vad/
[doc-url]: https://docs.rs/firered-vad
[crates-url]: https://crates.io/crates/firered-vad
