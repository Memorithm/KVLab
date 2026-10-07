#!/usr/bin/env bash
set -euo pipefail

# Dell T430 BKV-K4 true dual-socket local-shard gate.
#
# A monolithic deterministic corpus is compared with two disjoint page ranges:
#   shard 0 -> socket/node 0 local memory
#   shard 1 -> socket/node 1 local memory
# Both shards use the exact same query and contiguous global page-ID space.
# The run fails unless the concatenated shard candidate IDs exactly match the
# monolithic oracle candidate IDs.

TOTAL_PAGES="${1:-32000000}"
OUT="${2:-t430-k4-distinct-shards}"
SIGNATURE_BITS="${SIGNATURE_BITS:-256}"
MAX_DISTANCE="${MAX_DISTANCE:-96}"
WARMUP="${WARMUP:-5}"
REPETITIONS="${REPETITIONS:-25}"
SEED="${SEED:-1}"
WORKERS_PER_SOCKET="${WORKERS_PER_SOCKET:-16}"
MONOLITHIC_WORKERS="${MONOLITHIC_WORKERS:-32}"

if (( TOTAL_PAGES <= 1 || TOTAL_PAGES % 2 != 0 )); then
  echo "error: TOTAL_PAGES must be an even integer greater than one" >&2
  exit 1
fi
if (( REPETITIONS < 3 )); then
  echo "error: REPETITIONS must be at least three for an alternated campaign" >&2
  exit 1
fi
SHARD_PAGES=$((TOTAL_PAGES / 2))

NODE0_CPUS="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30"
NODE1_CPUS="1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if ! command -v numactl >/dev/null 2>&1; then
  echo "error: numactl is required" >&2
  exit 1
fi

TRACKED_DIRTY=false
if ! git diff --quiet || ! git diff --cached --quiet; then
  TRACKED_DIRTY=true
fi
if [[ "$TRACKED_DIRTY" == true ]]; then
  echo "error: tracked worktree changes would invalidate performance provenance" >&2
  exit 1
fi
PREEXISTING_UNTRACKED="$(git ls-files --others --exclude-standard | wc -l | tr -d ' ')"
HEAD="$(git rev-parse HEAD)"

TOPOLOGY_CSV="$(mktemp)"
trap 'rm -f "$TOPOLOGY_CSV"' EXIT
lscpu -p=CPU,CORE,SOCKET,NODE > "$TOPOLOGY_CSV"
python3 - "$TOPOLOGY_CSV" "$NODE0_CPUS" "$NODE1_CPUS" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = {
    0: [int(x) for x in sys.argv[2].split(",")],
    1: [int(x) for x in sys.argv[3].split(",")],
}
rows = {}
with path.open() as f:
    for raw in f:
        if raw.startswith("#") or not raw.strip():
            continue
        cpu, core, socket, node = (int(x) for x in raw.strip().split(","))
        rows[cpu] = (core, socket, node)

if set(socket for _, socket, _ in rows.values()) != {0, 1}:
    raise SystemExit("unexpected socket topology: expected sockets 0 and 1")
if set(node for _, _, node in rows.values()) != {0, 1}:
    raise SystemExit("unexpected NUMA topology: expected nodes 0 and 1")
if len({(socket, core) for core, socket, _ in rows.values()}) != 32:
    raise SystemExit("unexpected physical-core count: expected 32")

for node, cpus in expected.items():
    cores = set()
    for cpu in cpus:
        if cpu not in rows:
            raise SystemExit(f"expected CPU {cpu} is not online")
        core, socket, actual_node = rows[cpu]
        if socket != node or actual_node != node:
            raise SystemExit(
                f"CPU {cpu} maps to socket={socket}, node={actual_node}; "
                f"expected socket=node={node}"
            )
        cores.add(core)
    if len(cores) != 16:
        raise SystemExit(
            f"node {node} representative set spans {len(cores)} cores; expected 16"
        )
PY

if [[ -e "$OUT" ]]; then
  echo "error: output path already exists: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT/samples"
{
  echo "schema=bkv-k4-distinct-dual-local-campaign/v2"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$HEAD"
  echo "git_tracked_dirty=$TRACKED_DIRTY"
  echo "preexisting_untracked_count=$PREEXISTING_UNTRACKED"
  echo "corpus_mode=distinct-contiguous-global-page-ranges"
  echo "total_pages=$TOTAL_PAGES"
  echo "shard_pages=$SHARD_PAGES"
  echo "signature_bits=$SIGNATURE_BITS"
  echo "max_distance=$MAX_DISTANCE"
  echo "workers_per_socket=$WORKERS_PER_SOCKET"
  echo "monolithic_workers=$MONOLITHIC_WORKERS"
  echo "warmup=$WARMUP"
  echo "repetitions=$REPETITIONS"
  echo "seed=$SEED"
  echo "latency_method=paired-process-wall-clock-including-merge"
  echo "execution_order=alternated"
  echo "shard_launch_order=alternated"
  echo "historical_v1_pair_metric=estimate-only"
} > "$OUT/manifest.txt"

lscpu > "$OUT/lscpu.txt"
cp "$TOPOLOGY_CSV" "$OUT/lscpu-topology.csv"
numactl --hardware > "$OUT/numactl-hardware.txt"
cat /proc/meminfo > "$OUT/meminfo.txt"

cargo build --locked --release --manifest-path rust/bkv_scan/Cargo.toml --bins
BIN="$ROOT/rust/bkv_scan/target/release/bkv_distinct_shard_bench"

samples_csv="$OUT/paired-samples.csv"
printf '%s\n' \
  'repetition,execution_order,shard_launch_order,monolithic_e2e_ns,shard0_scan_ns,shard1_scan_ns,pair_scan_wall_ns,merge_ns,pair_e2e_ns,selected_pages,candidate_equality' \
  > "$samples_csv"

csv_value() {
  local path="$1"
  local field="$2"
  python3 - "$path" "$field" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
field = sys.argv[2]
with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 1:
    raise SystemExit(f"expected one row in {path}, got {len(rows)}")
print(rows[0][field])
PY
}

run_monolithic() {
  local sample_dir="$1"
  local start_ns end_ns

  start_ns="$(date +%s%N)"
  numactl --physcpubind=0-31 --interleave=0,1 \
    "$BIN" "$TOTAL_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
    0 "$TOTAL_PAGES" "$MONOLITHIC_WORKERS" "$WARMUP" 1 "$SEED" \
    "$sample_dir/monolithic.ids" > "$sample_dir/monolithic.csv"
  end_ns="$(date +%s%N)"
  monolithic_e2e_ns=$((end_ns - start_ns))
  if ! cmp -s "$OUT/oracle.ids" "$sample_dir/monolithic.ids"; then
    echo "error: timed monolithic candidates differ from frozen campaign oracle" >&2
    exit 1
  fi
}

run_dual_pair() {
  local sample_dir="$1"
  local launch_order="$2"
  local start_ns scan_end_ns merge_end_ns pid0 pid1 rc0=0 rc1=0

  start_ns="$(date +%s%N)"
  if [[ "$launch_order" == "shard0-first" ]]; then
    numactl --physcpubind="$NODE0_CPUS" --membind=0 \
      "$BIN" "$TOTAL_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
      0 "$SHARD_PAGES" "$WORKERS_PER_SOCKET" "$WARMUP" 1 "$SEED" \
      "$sample_dir/shard0.ids" > "$sample_dir/shard0.csv" &
    pid0=$!
    numactl --physcpubind="$NODE1_CPUS" --membind=1 \
      "$BIN" "$TOTAL_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
      "$SHARD_PAGES" "$SHARD_PAGES" "$WORKERS_PER_SOCKET" "$WARMUP" 1 "$SEED" \
      "$sample_dir/shard1.ids" > "$sample_dir/shard1.csv" &
    pid1=$!
  else
    numactl --physcpubind="$NODE1_CPUS" --membind=1 \
      "$BIN" "$TOTAL_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
      "$SHARD_PAGES" "$SHARD_PAGES" "$WORKERS_PER_SOCKET" "$WARMUP" 1 "$SEED" \
      "$sample_dir/shard1.ids" > "$sample_dir/shard1.csv" &
    pid1=$!
    numactl --physcpubind="$NODE0_CPUS" --membind=0 \
      "$BIN" "$TOTAL_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
      0 "$SHARD_PAGES" "$WORKERS_PER_SOCKET" "$WARMUP" 1 "$SEED" \
      "$sample_dir/shard0.ids" > "$sample_dir/shard0.csv" &
    pid0=$!
  fi

  wait "$pid0" || rc0=$?
  wait "$pid1" || rc1=$?
  if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
    echo "error: distinct-shard child benchmark failed: shard0=$rc0 shard1=$rc1" >&2
    exit 1
  fi
  scan_end_ns="$(date +%s%N)"

  cat "$sample_dir/shard0.ids" "$sample_dir/shard1.ids" > "$sample_dir/merged.ids"
  if ! cmp -s "$OUT/oracle.ids" "$sample_dir/merged.ids"; then
    echo "error: merged dual-local candidate IDs differ from monolithic oracle" >&2
    diff -u "$OUT/oracle.ids" "$sample_dir/merged.ids" | head -200 >&2 || true
    exit 1
  fi
  merge_end_ns="$(date +%s%N)"
  pair_scan_wall_ns=$((scan_end_ns - start_ns))
  merge_ns=$((merge_end_ns - scan_end_ns))
  pair_e2e_ns=$((merge_end_ns - start_ns))
}

# Freeze one untimed correctness oracle before alternating timed order. It is
# not included in either latency distribution.
numactl --physcpubind=0-31 --interleave=0,1 \
  "$BIN" "$TOTAL_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
  0 "$TOTAL_PAGES" "$MONOLITHIC_WORKERS" 0 1 "$SEED" \
  "$OUT/oracle.ids" > "$OUT/oracle.csv"
sha256sum "$OUT/oracle.ids" > "$OUT/oracle.ids.sha256"

for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  sample_dir="$OUT/samples/$(printf '%03d' "$repetition")"
  mkdir "$sample_dir"
  if (( repetition % 2 == 1 )); then
    execution_order="monolithic-first"
    shard_launch_order="shard0-first"
    echo "repetition $repetition/$REPETITIONS: monolithic then dual pair" >&2
    run_monolithic "$sample_dir"
    run_dual_pair "$sample_dir" "$shard_launch_order"
  else
    execution_order="dual-first"
    shard_launch_order="shard1-first"
    echo "repetition $repetition/$REPETITIONS: dual pair then monolithic" >&2
    run_dual_pair "$sample_dir" "$shard_launch_order"
    run_monolithic "$sample_dir"
  fi

  shard0_scan_ns="$(csv_value "$sample_dir/shard0.csv" median_ns)"
  shard1_scan_ns="$(csv_value "$sample_dir/shard1.csv" median_ns)"
  selected_pages="$(csv_value "$sample_dir/monolithic.csv" selected_pages)"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,exact\n' \
    "$repetition" "$execution_order" "$shard_launch_order" \
    "$monolithic_e2e_ns" "$shard0_scan_ns" "$shard1_scan_ns" \
    "$pair_scan_wall_ns" "$merge_ns" "$pair_e2e_ns" "$selected_pages" \
    >> "$samples_csv"
done

python3 -m kvlab.t430_pair_summary \
  "$samples_csv" "$OUT/summary.csv" "$TOTAL_PAGES" "$SIGNATURE_BITS"
sha256sum \
  "$OUT/manifest.txt" \
  "$OUT/oracle.ids" \
  "$OUT/paired-samples.csv" \
  "$OUT/summary.csv" > "$OUT/SHA256SUMS"
cat "$OUT/summary.csv"
