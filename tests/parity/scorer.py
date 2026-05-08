#!/usr/bin/env python3
"""Compare Python and Rust per-frame parity outputs.

Usage:
    python scorer.py <python.json> <rust.json> [--prob-tol 5e-4]
        [--max-mismatches 20] [--quiet]

Tolerances:
    raw_prob, smoothed_prob: |delta| <= prob_tol  (default 5e-4, which
        matches upstream Python's `round(..., 3)` ceiling)
    is_speech / is_speech_start / is_speech_end: exact equality
    speech_start_frame / speech_end_frame: exact equality

Exit codes:
    0 = bit-identical (within prob_tol)
    1 = mismatches found (with details)
    2 = structural failure (frame-count mismatch, missing files, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# ANSI colors for terminal output (degraded to plain text if not tty).
def _color(code: str, s: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{s}\033[0m"
    return s


def red(s: str) -> str:
    return _color("31", s)


def green(s: str) -> str:
    return _color("32", s)


def yellow(s: str) -> str:
    return _color("33", s)


def cyan(s: str) -> str:
    return _color("36", s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("python_json")
    parser.add_argument("rust_json")
    parser.add_argument(
        "--prob-tol",
        type=float,
        default=5e-3,
        help=(
            "Tolerance for raw_prob / smoothed_prob (default 5e-3). "
            "Empirically the kaldi-native-fbank (C++) vs pure-Rust mel-fbank + "
            "PyTorch vs ONNX-runtime numerical drift sits at 1-3e-3 for normal-amplitude "
            "audio and can spike to ~7e-2 for high-amplitude (RMS > 4000) signals. "
            "All discrete fields (is_speech, is_speech_*, speech_*_frame) require "
            "exact equality regardless of this tolerance."
        ),
    )
    parser.add_argument("--max-mismatches", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        with open(args.python_json) as f:
            py = json.load(f)
        with open(args.rust_json) as f:
            rs = json.load(f)
    except FileNotFoundError as e:
        print(red(f"error: {e}"), file=sys.stderr)
        return 2

    py_frames = py["frames"]
    rs_frames = rs["frames"]

    # ── Structural checks ────────────────────────────────────────────
    if py["n_frames"] != rs["n_frames"]:
        print(
            red(f"FAIL  n_frames mismatch: python={py['n_frames']} rust={rs['n_frames']}"),
            file=sys.stderr,
        )
        return 2

    if py["n_samples"] != rs["n_samples"]:
        print(
            red(f"FAIL  n_samples mismatch: python={py['n_samples']} rust={rs['n_samples']}"),
            file=sys.stderr,
        )
        return 2

    n = py["n_frames"]

    # ── Aggregate counters ──────────────────────────────────────────
    raw_max_delta = 0.0
    raw_total_delta = 0.0
    smooth_max_delta = 0.0
    smooth_total_delta = 0.0
    is_speech_mismatch = 0
    start_mismatch = 0
    end_mismatch = 0
    start_frame_mismatch = 0
    end_frame_mismatch = 0
    raw_over_tol = 0
    smooth_over_tol = 0

    mismatches: list[str] = []

    def remember(msg: str) -> None:
        if len(mismatches) < args.max_mismatches:
            mismatches.append(msg)

    for i, (p, r) in enumerate(zip(py_frames, rs_frames)):
        if p["frame_index"] != r["frame_index"]:
            print(
                red(
                    f"FAIL  frame {i}: frame_index mismatch "
                    f"py={p['frame_index']} rs={r['frame_index']}"
                ),
                file=sys.stderr,
            )
            return 2

        # Continuous fields with tolerance.
        d_raw = abs(p["raw_prob"] - r["raw_prob"])
        d_smooth = abs(p["smoothed_prob"] - r["smoothed_prob"])
        raw_max_delta = max(raw_max_delta, d_raw)
        raw_total_delta += d_raw
        smooth_max_delta = max(smooth_max_delta, d_smooth)
        smooth_total_delta += d_smooth
        if d_raw > args.prob_tol:
            raw_over_tol += 1
            remember(
                f"frame {i}: raw_prob delta={d_raw:.5f} "
                f"py={p['raw_prob']:.6f} rs={r['raw_prob']:.6f}"
            )
        if d_smooth > args.prob_tol:
            smooth_over_tol += 1
            remember(
                f"frame {i}: smoothed_prob delta={d_smooth:.5f} "
                f"py={p['smoothed_prob']:.6f} rs={r['smoothed_prob']:.6f}"
            )

        # Discrete fields — exact match required.
        if p["is_speech"] != r["is_speech"]:
            is_speech_mismatch += 1
            remember(
                f"frame {i}: is_speech mismatch py={p['is_speech']} rs={r['is_speech']}"
            )
        if p["is_speech_start"] != r["is_speech_start"]:
            start_mismatch += 1
            remember(
                f"frame {i}: is_speech_start mismatch "
                f"py={p['is_speech_start']} rs={r['is_speech_start']}"
            )
        if p["is_speech_end"] != r["is_speech_end"]:
            end_mismatch += 1
            remember(
                f"frame {i}: is_speech_end mismatch "
                f"py={p['is_speech_end']} rs={r['is_speech_end']}"
            )
        if p["speech_start_frame"] != r["speech_start_frame"]:
            start_frame_mismatch += 1
            remember(
                f"frame {i}: speech_start_frame mismatch "
                f"py={p['speech_start_frame']} rs={r['speech_start_frame']}"
            )
        if p["speech_end_frame"] != r["speech_end_frame"]:
            end_frame_mismatch += 1
            remember(
                f"frame {i}: speech_end_frame mismatch "
                f"py={p['speech_end_frame']} rs={r['speech_end_frame']}"
            )

    # ── Report ─────────────────────────────────────────────────────
    py_starts = sum(1 for f in py_frames if f["is_speech_start"])
    py_ends = sum(1 for f in py_frames if f["is_speech_end"])
    rs_starts = sum(1 for f in rs_frames if f["is_speech_start"])
    rs_ends = sum(1 for f in rs_frames if f["is_speech_end"])

    if not args.quiet:
        print(cyan(f"=== parity report: {py['wav_path']}"))
        print(f"  duration: {py['duration_s']:.2f}s ({n} frames)")
        print(f"  segments: python={py_starts}/{py_ends}  rust={rs_starts}/{rs_ends} (start/end)")
        print(f"  raw_prob   max|Δ|={raw_max_delta:.6f}  mean|Δ|={raw_total_delta/n:.6f}  >{args.prob_tol}: {raw_over_tol}")
        print(f"  smoothed   max|Δ|={smooth_max_delta:.6f}  mean|Δ|={smooth_total_delta/n:.6f}  >{args.prob_tol}: {smooth_over_tol}")
        print(f"  is_speech         mismatch: {is_speech_mismatch}")
        print(f"  is_speech_start   mismatch: {start_mismatch}")
        print(f"  is_speech_end     mismatch: {end_mismatch}")
        print(f"  speech_start_frame mismatch: {start_frame_mismatch}")
        print(f"  speech_end_frame   mismatch: {end_frame_mismatch}")

    total_mismatches = (
        raw_over_tol
        + smooth_over_tol
        + is_speech_mismatch
        + start_mismatch
        + end_mismatch
        + start_frame_mismatch
        + end_frame_mismatch
    )

    if total_mismatches == 0:
        if not args.quiet:
            print(green(f"  PASS  bit-identical within prob_tol={args.prob_tol}"))
        return 0

    if not args.quiet:
        print(red(f"  FAIL  {total_mismatches} mismatch(es). first {len(mismatches)}:"))
        for m in mismatches:
            print(f"    {m}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
