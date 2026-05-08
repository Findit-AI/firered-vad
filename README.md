# firered-vad

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

## Quick start

```rust,no_run
use firered_vad::Vad;

fn main() -> firered_vad::Result<()> {
    let pcm: Vec<f32> = vec![0.0; 16_000]; // 16 kHz f32 PCM in [-1.0, 1.0]
    let mut vad = Vad::bundled()?;

    for chunk in pcm.chunks(1_600) {
        let mut chunk: &[f32] = chunk;
        while let Some(segment) = vad.push_samples(chunk)? {
            // Slice the original PCM to recover the speech window.
            let _speech = &pcm[segment.range_usize()];
            // ... feed `speech` into Whisper / your transcriber.
            chunk = &[]; // drain remaining buffered segments before pushing the next chunk
        }
    }
    if let Some(segment) = vad.finish()? {
        // Trailing segment (open at end-of-stream).
        let _speech = &pcm[segment.range_usize()];
    }
    Ok(())
}
```

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
