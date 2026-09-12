#!/usr/bin/env bash
set -euo pipefail

# Reproducible Dell T430 BKV-K4 large-memory sweep.
# This runner is intentionally hardware-specific. It relies on the verified
# T430 topology captured on 2026-09-12:
#   node 0 physical-thread representatives: 0,2,...,30
#   node 1 physical-thread representatives: 1,3,...,31
#   SMT siblings: 32-63

PAGES="${1:-32000000}"
OUT="${2:-t430-k4-32m}"
SIGNATURE_BITS="${SIGNATURE_BITS:-256}"
MAX_DISTANCE="${MAX_DISTANCE:-96}"
WARMUP="${WARMUP:-5}"
REPETITIONS="${REPETITIONS:-25}"
SEED="${SEED:-1}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
mkdir -p "$OUT"

if ! command -v numactl >/dev/null 2>&1; then
  echo "error: numactl is required" >&2
  exit 1
fi

HEAD="$(git rev-parse HEAD)"
{
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$HEAD"
  echo "git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)"
  echo "pages=$PAGES"
  echo "signature_bits=$SIGNATURE_BITS"
  echo "max_distance=$MAX_DISTANCE"
  echo "warmup=$WARMUP"
  echo "repetitions=$REPETITIONS"
  echo "seed=$SEED"
} > "$OUT/manifest.txt"

lscpu > "$OUT/lscpu.txt"
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE > "$OUT/lscpu-topology.txt"
numactl --hardware > "$OUT/numactl-hardware.txt"
cat /proc/meminfo > "$OUT/meminfo.txt"

cargo build --release --manifest-path rust/bkv_scan/Cargo.toml --bins
BIN="$ROOT/rust/bkv_scan/target/release/bkv_flat_scan_bench"
SUM="$ROOT/rust/bkv_scan/target/release/bkv_scaling_summary"

run_local_node0() {
  local label="$1"
  local cpus="$2"
  local workers="$3"
  echo "running $label: workers=$workers cpus=$cpus membind=0" >&2
  numactl --physcpubind="$cpus" --membind=0 \
    "$BIN" "$PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
    "$workers" "$WARMUP" "$REPETITIONS" "$SEED" \
    > "$OUT/$label.csv"
}

run_interleaved() {
  local label="$1"
  local cpus="$2"
  local workers="$3"
  echo "running $label: workers=$workers cpus=$cpus interleave=0,1" >&2
  numactl --physcpubind="$cpus" --interleave=0,1 \
    "$BIN" "$PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
    "$workers" "$WARMUP" "$REPETITIONS" "$SEED" \
    > "$OUT/$label.csv"
}

run_local_node0 w1 0 1
run_local_node0 w8 0,2,4,6,8,10,12,14 8
run_local_node0 w16 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 16
run_interleaved w32 0-31 32
run_interleaved w64-smt 0-63 64

"$SUM" \
  "$OUT/w1.csv" \
  "$OUT/w8.csv" \
  "$OUT/w16.csv" \
  "$OUT/w32.csv" \
  "$OUT/w64-smt.csv" \
  > "$OUT/scaling.csv"

cat "$OUT/scaling.csv"
