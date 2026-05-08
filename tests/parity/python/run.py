#!/usr/bin/env python3
"""Run upstream FireRedStreamVad on a 16 kHz mono WAV and dump per-frame results as JSON.

Output shape matches the Rust runner's format byte-for-byte modulo
JSON formatting. The scorer (../scorer.py) consumes both.

Usage:
    python run.py --model-dir <dir> --wav <path> --out <json>
        [--smooth-window-size 5] [--speech-threshold 0.5] [--pad-start-frame 5]
        [--min-speech-frame 8] [--max-speech-frame 2000] [--min-silence-frame 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from fireredvad import FireRedStreamVad, FireRedStreamVadConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Path to FireRedVAD/Stream-VAD")
    parser.add_argument("--wav", required=True, help="Path to 16 kHz mono WAV")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--smooth-window-size", type=int, default=5)
    parser.add_argument("--speech-threshold", type=float, default=0.5)
    parser.add_argument("--pad-start-frame", type=int, default=5)
    parser.add_argument("--min-speech-frame", type=int, default=8)
    parser.add_argument("--max-speech-frame", type=int, default=2000)
    parser.add_argument("--min-silence-frame", type=int, default=20)
    args = parser.parse_args()

    cfg = FireRedStreamVadConfig(
        use_gpu=False,
        smooth_window_size=args.smooth_window_size,
        speech_threshold=args.speech_threshold,
        pad_start_frame=args.pad_start_frame,
        min_speech_frame=args.min_speech_frame,
        max_speech_frame=args.max_speech_frame,
        min_silence_frame=args.min_silence_frame,
    )
    vad = FireRedStreamVad.from_pretrained(args.model_dir, cfg)

    # Monkey-patch the postprocessor's StreamVadFrameResult creation to
    # NOT round raw_prob and smoothed_prob — bit-parity comparison is
    # impossible against rounded values (5e-4 ceiling). We grab the
    # unrounded values via a small wrapper around `process_one_frame`.
    orig_process = vad.postprocessor.process_one_frame

    def process_one_frame_unrounded(raw_prob: float):
        # Mirror upstream's logic but capture unrounded raw and smoothed.
        # Implemented by calling the original AND capturing the local
        # `smoothed_prob` via reproducing the smoothing in lockstep.
        # (Simpler: shadow the rounding by replacing it in the result.)
        result = orig_process(raw_prob)
        # Re-compute smoothed_prob from our shadow window so we can store it
        # unrounded alongside the existing (rounded) `result.smoothed_prob`.
        # We can't reach the postprocessor's internal smooth_window from
        # here without monkey-patching, so instead we just expose raw_prob
        # unrounded and re-run smoothing externally below.
        result._unrounded_raw_prob = float(raw_prob)
        return result

    vad.postprocessor.process_one_frame = process_one_frame_unrounded

    wav, sr = sf.read(args.wav, dtype="int16")
    if sr != 16000:
        print(f"error: expected 16 kHz, got {sr}", file=sys.stderr)
        return 1
    if wav.ndim != 1:
        print(f"error: expected mono, got shape {wav.shape}", file=sys.stderr)
        return 1

    results = vad.detect_chunk(wav)

    # Re-compute smoothed_prob OURSELVES from the captured unrounded raw
    # probs, using upstream's exact algorithm but skipping the round().
    from collections import deque

    smooth_size = max(1, args.smooth_window_size)
    smooth_window: deque[float] = deque()
    smooth_sum: float = 0.0

    def compute_smoothed(raw: float) -> float:
        nonlocal smooth_sum
        if smooth_size <= 1:
            return raw
        smooth_window.append(raw)
        smooth_sum += raw
        if len(smooth_window) > smooth_size:
            smooth_sum -= smooth_window.popleft()
        return smooth_sum / len(smooth_window)

    frames = []
    for r in results:
        frame_index = int(r.frame_idx) - 1  # upstream is 1-based; we shift
        speech_start_frame = (
            int(r.speech_start_frame) - 1 if r.speech_start_frame > 0 else None
        )
        speech_end_frame = (
            int(r.speech_end_frame) - 1 if r.speech_end_frame > 0 else None
        )
        raw_prob_unrounded = float(getattr(r, "_unrounded_raw_prob", r.raw_prob))
        smoothed_unrounded = compute_smoothed(raw_prob_unrounded)
        frames.append(
            {
                "frame_index": frame_index,
                "raw_prob": raw_prob_unrounded,
                "smoothed_prob": smoothed_unrounded,
                "is_speech": bool(r.is_speech),
                "is_speech_start": bool(r.is_speech_start),
                "is_speech_end": bool(r.is_speech_end),
                "speech_start_frame": speech_start_frame,
                "speech_end_frame": speech_end_frame,
            }
        )

    output = {
        "wav_path": str(Path(args.wav).resolve()),
        "sample_rate": sr,
        "n_samples": int(wav.shape[0]),
        "duration_s": round(wav.shape[0] / sr, 3),
        "n_frames": len(frames),
        "config": {
            "smooth_window_size": args.smooth_window_size,
            "speech_threshold": args.speech_threshold,
            "pad_start_frame": args.pad_start_frame,
            "min_speech_frame": args.min_speech_frame,
            "max_speech_frame": args.max_speech_frame,
            "min_silence_frame": args.min_silence_frame,
        },
        "frames": frames,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=None, separators=(",", ":"))

    print(
        f"python: {args.wav} -> {args.out}  "
        f"({len(frames)} frames, "
        f"{sum(1 for f in frames if f['is_speech_start'])} starts, "
        f"{sum(1 for f in frames if f['is_speech_end'])} ends)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
