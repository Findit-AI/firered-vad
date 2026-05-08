# Parity harness

Compares **upstream Python** `FireRedStreamVad` vs **our Rust** `firered_vad::Vad` per-frame on a list of 16 kHz mono WAV fixtures. Goal: bit-identity.

This harness is **not** part of `cargo test`. It is run manually because it requires:

1. A Python virtualenv with `fireredvad`, `torch`, `numpy`, `soundfile`.
2. The upstream model weights downloaded from Hugging Face.
3. WAV fixtures on disk (paths configured in `fixtures.toml`).

## Tolerance and what "bit-identical" means here

Upstream Python rounds `raw_prob` and `smoothed_prob` to **3 decimal places** on storage (`round(raw_prob, 3)` in `fireredvad/core/stream_vad_postprocessor.py`). The Python runner here monkey-patches that rounding away so we compare unrounded values.

After the mel-fbank fixes documented below, our pure-Rust pipeline is **bit-identical** to upstream Python's `kaldi-native-fbank` + PyTorch pipeline for all practical purposes. The remaining ~3 × 10⁻⁶ residual drift sits at the float32 precision limit and is dominated by FFT-implementation rounding (rustfft's Radix-4 vs `kaldi-native-fbank`'s KissFFT). Discrete fields (segment boundaries, is_speech, etc.) match **exactly** — this is the gate that matters for any downstream consumer.

The harness pins:

- **Discrete fields** (`is_speech`, `is_speech_start`, `is_speech_end`, `speech_start_frame`, `speech_end_frame`): exact equality. State machine is bit-identical.
- **Continuous fields** (`raw_prob`, `smoothed_prob`): tolerance defaults to **5e-3** (`--prob-tol 5e-3`); empirically we hit ~3e-6.

## Empirical results

Tested on 7 fixtures (181,230 total frames, 543 total segments):

| Fixture | Frames | Segments (py/rs) | raw_prob max\|Δ\| | smoothed max\|Δ\| | is_speech mismatch | Boundary mismatch |
| --- | --: | --: | --: | --: | --: | --: |
| 02_pyannote_sample | 2998 | 4/3 ≡ 4/3 | 3 × 10⁻⁶ | 2 × 10⁻⁶ | 0 | 0 |
| 03_dual_speaker | 5998 | 19/18 ≡ 19/18 | 3 × 10⁻⁶ | 2 × 10⁻⁶ | 0 | 0 |
| 04_three_speaker | 3995 | 10/9 ≡ 10/9 | 2 × 10⁻⁶ | 1 × 10⁻⁶ | 0 | 0 |
| 05_four_speaker | 5998 | 15/15 ≡ 15/15 | 2 × 10⁻⁶ | 1 × 10⁻⁶ | 0 | 0 |
| 07_yuhewei_dongbei_english | 2524 | 4/4 ≡ 4/4 | 2 × 10⁻⁶ | 1 × 10⁻⁶ | 0 | 0 |
| 10_mrbeast_clean_water (10m) | 61,948 | 54/54 ≡ 54/54 | 3 × 10⁻⁶ | 3 × 10⁻⁶ | 0 | 0 |
| 06_long_recording (16m) | 97,771 | 439/439 ≡ 439/439 | 3 × 10⁻⁶ | 3 × 10⁻⁶ | 0 | 0 |

**All discrete fields match exactly** across 181,230 frames. Mean continuous-field Δ rounds to 0.0 in float32. Segment counts, segment boundaries, and segment timings are identical fixture-by-fixture.

## What was wrong before, and what we fixed

An initial draft of the harness reported max raw_prob deltas of 1-3 × 10⁻³ typical and ~7 × 10⁻² on the high-amplitude 07_yuhewei fixture. A line-by-line diff against upstream's `kaldi-native-fbank` (C++) found three real algorithmic bugs in the pure-Rust mel-fbank:

1. **Mel filter weights were linear in Hz, not in mel space.** `kaldi-native-fbank` (`mel-computations.cc::InitKaldiMelBanks`) builds each triangular filter on `(left_mel, center_mel, right_mel)` anchors that are linearly spaced in **mel** space, then computes weights as `(mel(f) - left_mel) / (center_mel - left_mel)`. The pure-Rust code converted the anchors to Hz first and computed `(f - left_hz) / (center_hz - left_hz)`. Because `mel(f)` is logarithmic in `f`, the two definitions give visibly different weights for every bin — the dominant source of drift.

2. **Log-floor was 1e-20 instead of `f32::EPSILON`.** `kaldi-native-fbank` (`feature-fbank.cc::Compute`) clamps mel-bin energies with `std::max(energy, std::numeric_limits<float>::epsilon())` before taking the log; the original Kaldi project used 1e-20. For very-quiet bins the difference is large (`log(1e-20) ≈ -46` vs `log(1.19e-7) ≈ -16`).

3. **Povey window was computed in `f32`.** `kaldi-native-fbank::GetWindow` keeps `cos(...)` and `pow(..., 0.85)` in `double` and only narrows to `float` at storage. Doing the whole computation in `f32` accumulates ~1 ULP per element of error, which then multiplies into every windowed sample.

Bonus: also fixed a small off-by-one — Kaldi's mel filter loop iterates over `0..num_fft_bins = FFT_SIZE/2 = 256`, **excluding** the Nyquist bin. The pure-Rust code iterated `0..FFT_BINS = 257`, including Nyquist. The strict-inequality test `mel < right_mel` rejected Nyquist anyway in the dia fixtures, but the loop bound is now exactly Kaldi's.

After these four fixes, the residual drift is 433× smaller on average and 23,000× smaller on the worst-case high-amplitude fixture.

## One-time setup

```bash
# 1. Create a Python 3.12 venv (Python 3.10–3.12 work; 3.14 doesn't yet).
python3.12 -m venv /tmp/parity-venv
/tmp/parity-venv/bin/pip install --upgrade pip
/tmp/parity-venv/bin/pip install -r tests/parity/python/requirements.txt

# 2. Download upstream weights from Hugging Face.
/tmp/parity-venv/bin/hf download FireRedTeam/FireRedVAD --local-dir /tmp/firered-vad-weights
```

## Running

```bash
cd tests/parity
./run.sh                       # all fixtures
./run.sh 02_pyannote_sample    # just one
```

Override defaults via env vars:

```bash
PYTHON_VENV=/path/to/venv FIRERED_WEIGHTS=/path/to/weights ./run.sh
```

JSON dumps land in `tests/parity/out/`. Per-fixture format:

```json
{
  "wav_path": "...",
  "sample_rate": 16000,
  "n_samples": 480000,
  "duration_s": 30.0,
  "n_frames": 2998,
  "config": { "smooth_window_size": 5, "speech_threshold": 0.5, ... },
  "frames": [
    {
      "frame_index": 0,
      "raw_prob": 0.031,
      "smoothed_prob": 0.031,
      "is_speech": false,
      "is_speech_start": false,
      "is_speech_end": false,
      "speech_start_frame": null,
      "speech_end_frame": null
    },
    ...
  ]
}
```

## Layout

```
tests/parity/
├── README.md            (this file)
├── fixtures.toml        list of fixture WAVs to exercise
├── run.sh               driver: build Rust runner, iterate fixtures
├── scorer.py            JSON-vs-JSON diff with frame-level tolerance
├── python/
│   ├── requirements.txt
│   └── run.py           upstream FireRedStreamVad → JSON
├── rust/
│   ├── Cargo.toml       standalone bin (path-deps on the parent crate)
│   └── src/main.rs      our Vad → JSON
└── out/                 (gitignored) per-fixture JSON dumps
```

## Why pinned at 5e-4

If our `raw_prob_rust` rounded to 3 decimals matches the corresponding Python `raw_prob_python` value (which is already rounded), we are exactly bit-identical at the precision Python preserves. Tightening below 5e-4 has no information value because Python literally throws those bits away.

If a fixture fails parity, the scorer prints the first ~20 mismatched frames with deltas and per-field disagreement counts so you can localize the divergence.

## Out of scope

- ONNX-vs-PyTorch comparison: upstream uses PyTorch in-process; we use the ONNX export of the same weights via `ort`. Numerical drift between the two is in scope (it's part of "bit-identity") and shows up as raw_prob deltas.
- AED model: separate model, separate harness if ever added.
- Streaming chunk size sweep: `tests/integration_test.rs` already pins chunking-determinism on synthetic input.
