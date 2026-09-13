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

mkdir -p "$OUT"
{
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
} > "$OUT/manifest.txt"

lscpu > "$OUT/lscpu.txt"
cp "$TOPOLOGY_CSV" "$OUT/lscpu-topology.csv"
numactl --hardware > "$OUT/numactl-hardware.txt"
cat /proc/meminfo > "$OUT/meminfo.txt"

cargo build --release --manifest-path rust/bkv_scan/Cargo.toml --bins
BIN="$ROOT/rust/bkv_scan/target/release/bkv_distinct_shard_bench"

# Exact monolithic oracle over the complete deterministic page stream.
echo "running monolithic oracle" >&2
numactl --physcpubind=0-31 --interleave=0,1 \
  "$BIN" "$TOTAL_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
  0 "$TOTAL_PAGES" "$MONOLITHIC_WORKERS" "$WARMUP" "$REPETITIONS" "$SEED" \
  "$OUT/monolithic.ids" > "$OUT/monolithic.csv"

# True disjoint local shards, run concurrently. Shard 1 advances through the
# same deterministic page stream so its global IDs and signatures are exactly
# the second half of the monolithic corpus.
echo "running dual local distinct shards" >&2
numactl --physcpubind="$NODE0_CPUS" --membind=0 \
  "$BIN" "$TOTAL_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
  0 "$SHARD_PAGES" "$WORKERS_PER_SOCKET" "$WARMUP" "$REPETITIONS" "$SEED" \
  "$OUT/shard0.ids" > "$OUT/shard0.csv" &
pid0=$!

numactl --physcpubind="$NODE1_CPUS" --membind=1 \
  "$BIN" "$TOTAL_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
  "$SHARD_PAGES" "$SHARD_PAGES" "$WORKERS_PER_SOCKET" "$WARMUP" "$REPETITIONS" "$SEED" \
  "$OUT/shard1.ids" > "$OUT/shard1.csv" &
pid1=$!

rc=0
wait "$pid0" || rc=$?
wait "$pid1" || rc=$?
if [[ "$rc" -ne 0 ]]; then
  echo "error: distinct-shard child benchmark failed" >&2
  exit "$rc"
fi

cat "$OUT/shard0.ids" "$OUT/shard1.ids" > "$OUT/merged.ids"
if ! cmp -s "$OUT/monolithic.ids" "$OUT/merged.ids"; then
  echo "error: merged dual-local candidate IDs differ from monolithic oracle" >&2
  diff -u "$OUT/monolithic.ids" "$OUT/merged.ids" | head -200 >&2 || true
  exit 1
fi

echo "candidate equality: exact" >&2

python3 - "$OUT" "$TOTAL_PAGES" "$SIGNATURE_BITS" <<'PY'
import csv
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
total_pages = int(sys.argv[2])
signature_bits = int(sys.argv[3])


def load(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise SystemExit(f"expected one row in {path}, got {len(rows)}")
    row = rows[0]
    return {
        "median_ns": int(row["median_ns"]),
        "selected_pages": int(row["selected_pages"]),
        "page_offset": int(row["page_offset"]),
        "shard_pages": int(row["shard_pages"]),
    }

mono = load(out / "monolithic.csv")
s0 = load(out / "shard0.csv")
s1 = load(out / "shard1.csv")

if s0["page_offset"] != 0 or s1["page_offset"] != s0["shard_pages"]:
    raise SystemExit("shard page ranges are not contiguous")
if s0["shard_pages"] + s1["shard_pages"] != total_pages:
    raise SystemExit("shard page ranges do not cover total_pages")
if s0["selected_pages"] + s1["selected_pages"] != mono["selected_pages"]:
    raise SystemExit("selected-page count differs from monolithic oracle")

pair_ns = max(s0["median_ns"], s1["median_ns"])
logical_bytes = total_pages * signature_bits / 8
pages_per_second = total_pages / (pair_ns / 1e9)
logical_gb_s = logical_bytes / (pair_ns / 1e9) / 1e9
balance = min(s0["median_ns"], s1["median_ns"]) / pair_ns
speedup_vs_monolithic = mono["median_ns"] / pair_ns

with (out / "summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "schema",
        "monolithic_median_ns",
        "shard0_median_ns",
        "shard1_median_ns",
        "dual_pair_median_ns",
        "dual_pages_per_second",
        "dual_logical_gb_per_second",
        "socket_balance",
        "speedup_vs_monolithic_interleave",
        "selected_pages",
        "candidate_equality",
    ])
    w.writerow([
        "bkv-k4-distinct-dual-local-v1",
        mono["median_ns"],
        s0["median_ns"],
        s1["median_ns"],
        pair_ns,
        f"{pages_per_second:.6f}",
        f"{logical_gb_s:.6f}",
        f"{balance:.6f}",
        f"{speedup_vs_monolithic:.6f}",
        mono["selected_pages"],
        "exact",
    ])

print((out / "summary.csv").read_text(), end="")
PY
