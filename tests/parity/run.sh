#!/usr/bin/env bash
# Driver for the firered-vad parity harness.
#
# Compares per-frame outputs of upstream Python `FireRedStreamVad` vs
# our Rust `firered_vad::Vad` on a list of 16 kHz mono WAV fixtures.
#
# Required environment:
#   PYTHON_VENV       Path to a Python venv with `fireredvad`, `torch`,
#                     `numpy`, and `soundfile` installed.
#                     Default: /tmp/parity-venv
#   FIRERED_WEIGHTS   Path to upstream weights directory containing
#                     `Stream-VAD/{model.pth.tar, cmvn.ark}`.
#                     Default: /tmp/firered-vad-weights
#
# Usage:
#   ./run.sh                 # run all fixtures
#   ./run.sh <fixture-name>  # run just one (e.g. 02_pyannote_sample)
#
# Exit codes:
#   0  every fixture passed
#   1  one or more fixtures failed (mismatches)
#   2  structural failure (missing files, build error, etc.)

set -euo pipefail

cd "$(dirname "$0")"

PYTHON_VENV="${PYTHON_VENV:-/tmp/parity-venv}"
FIRERED_WEIGHTS="${FIRERED_WEIGHTS:-/tmp/firered-vad-weights}"
PYTHON="${PYTHON_VENV}/bin/python"
FIXTURE_FILTER="${1:-}"

OUT_DIR="out"
mkdir -p "${OUT_DIR}"

# ── Sanity checks ──────────────────────────────────────────────────
if [ ! -x "${PYTHON}" ]; then
  echo "error: Python venv not found at ${PYTHON_VENV}" >&2
  echo "       create one with: python3.12 -m venv ${PYTHON_VENV}" >&2
  echo "       then: ${PYTHON_VENV}/bin/pip install -r python/requirements.txt" >&2
  exit 2
fi
if [ ! -f "${FIRERED_WEIGHTS}/Stream-VAD/model.pth.tar" ]; then
  echo "error: upstream weights not found at ${FIRERED_WEIGHTS}/Stream-VAD/" >&2
  echo "       download via:" >&2
  echo "       ${PYTHON} -m hf download FireRedTeam/FireRedVAD --local-dir ${FIRERED_WEIGHTS}" >&2
  exit 2
fi

# ── Build Rust runner once ────────────────────────────────────────
echo "building rust parity runner..."
RUST_BIN="$(cd rust && cargo build --release --bin firered-vad-parity --message-format=json 2>/dev/null \
  | awk -F\" '/"executable":"[^"]+firered-vad-parity"/ {for (i=1;i<=NF;i++) if ($i=="executable") {print $(i+2); exit}}')"
if [ -z "${RUST_BIN}" ] || [ ! -x "${RUST_BIN}" ]; then
  echo "error: cargo build did not produce the firered-vad-parity executable" >&2
  exit 2
fi
echo "  bin: ${RUST_BIN}"

# ── Iterate fixtures ───────────────────────────────────────────────
PASS=0
FAIL=0
FAILED_NAMES=()

# Parse fixtures.toml entries (simple awk — keys "name" and "wav" per [[fixtures]] block)
mapfile -t FIXTURE_LINES < <(awk '
  /^\[\[fixtures\]\]/      { in_block=1; name=""; wav=""; next }
  in_block && /^name *= *"/ { sub(/^name *= *"/, ""); sub(/".*$/, ""); name=$0; next }
  in_block && /^wav *= *"/  { sub(/^wav *= *"/, "");  sub(/".*$/, ""); wav=$0; print name "\t" wav; in_block=0 }
' fixtures.toml)

for line in "${FIXTURE_LINES[@]}"; do
  name="${line%%$'\t'*}"
  wav="${line#*$'\t'}"

  if [ -n "${FIXTURE_FILTER}" ] && [ "${name}" != "${FIXTURE_FILTER}" ]; then
    continue
  fi

  if [ ! -f "${wav}" ]; then
    echo "warning: skipping ${name}: ${wav} not found" >&2
    continue
  fi

  echo
  echo "=== ${name} ==="

  py_json="${OUT_DIR}/${name}.python.json"
  rs_json="${OUT_DIR}/${name}.rust.json"

  "${PYTHON}" python/run.py \
    --model-dir "${FIRERED_WEIGHTS}/Stream-VAD" \
    --wav "${wav}" \
    --out "${py_json}"

  "${RUST_BIN}" --wav "${wav}" --out "${rs_json}"

  if "${PYTHON}" scorer.py "${py_json}" "${rs_json}"; then
    PASS=$((PASS + 1))
  else
    rc=$?
    FAIL=$((FAIL + 1))
    FAILED_NAMES+=("${name}")
    if [ "${rc}" -eq 2 ]; then
      echo "structural failure on ${name} — aborting" >&2
      exit 2
    fi
  fi
done

# ── Summary ────────────────────────────────────────────────────────
echo
echo "──────────────────────────────────"
echo "parity summary: ${PASS} pass, ${FAIL} fail"
if [ "${FAIL}" -gt 0 ]; then
  echo "failed fixtures:"
  for n in "${FAILED_NAMES[@]}"; do
    echo "  - ${n}"
  done
  exit 1
fi
echo "all fixtures bit-identical."
