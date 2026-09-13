#!/usr/bin/env bash
set -euo pipefail

# Dell T430 BKV-K4 dual-socket NUMA placement microbenchmark.
#
# Purpose: isolate memory-placement effects before introducing semantically
# distinct shard identities. Both sockets scan the same deterministic corpus
# and query concurrently. This mirrored corpus is intentional: local,
# interleaved, and remote modes therefore differ only in NUMA placement.
#
# Verified T430 physical-thread representatives:
#   node 0: 0,2,...,30
#   node 1: 1,3,...,31

SHARD_PAGES="${1:-16000000}"
OUT="${2:-t430-k4-dual-numa}"
SIGNATURE_BITS="${SIGNATURE_BITS:-256}"
MAX_DISTANCE="${MAX_DISTANCE:-96}"
WARMUP="${WARMUP:-5}"
REPETITIONS="${REPETITIONS:-25}"
SEED="${SEED:-1}"
WORKERS_PER_SOCKET="${WORKERS_PER_SOCKET:-16}"

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

# Fail closed if the physical-thread representatives no longer describe the
# verified two-socket/two-node T430 topology. `lscpu -p` is locale-independent.
TOPOLOGY_CSV="$(mktemp)"
trap 'rm -f "$TOPOLOGY_CSV"' EXIT
lscpu -p=CPU,CORE,SOCKET,NODE > "$TOPOLOGY_CSV"
python3 - "$TOPOLOGY_CSV" "$NODE0_CPUS" "$NODE1_CPUS" <<'PY'
import csv
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
            f"node {node} physical representative set spans {len(cores)} cores; expected 16"
        )
PY

mkdir -p "$OUT"
{
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_head=$HEAD"
  echo "git_tracked_dirty=$TRACKED_DIRTY"
  echo "preexisting_untracked_count=$PREEXISTING_UNTRACKED"
  echo "corpus_mode=mirrored-identical-pages-placement-microbenchmark"
  echo "shard_pages=$SHARD_PAGES"
  echo "total_logical_pages=$((SHARD_PAGES * 2))"
  echo "signature_bits=$SIGNATURE_BITS"
  echo "max_distance=$MAX_DISTANCE"
  echo "workers_per_socket=$WORKERS_PER_SOCKET"
  echo "warmup=$WARMUP"
  echo "repetitions=$REPETITIONS"
  echo "seed=$SEED"
} > "$OUT/manifest.txt"

lscpu > "$OUT/lscpu.txt"
cp "$TOPOLOGY_CSV" "$OUT/lscpu-topology.csv"
numactl --hardware > "$OUT/numactl-hardware.txt"
cat /proc/meminfo > "$OUT/meminfo.txt"

cargo build --release --manifest-path rust/bkv_scan/Cargo.toml --bins
BIN="$ROOT/rust/bkv_scan/target/release/bkv_flat_scan_bench"

run_pair() {
  local label="$1"
  local placement0="$2"
  local placement1="$3"

  echo "running $label" >&2

  numactl --physcpubind="$NODE0_CPUS" $placement0 \
    "$BIN" "$SHARD_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
    "$WORKERS_PER_SOCKET" "$WARMUP" "$REPETITIONS" "$SEED" \
    > "$OUT/${label}-node0.csv" &
  local pid0=$!

  numactl --physcpubind="$NODE1_CPUS" $placement1 \
    "$BIN" "$SHARD_PAGES" "$SIGNATURE_BITS" "$MAX_DISTANCE" \
    "$WORKERS_PER_SOCKET" "$WARMUP" "$REPETITIONS" "$SEED" \
    > "$OUT/${label}-node1.csv" &
  local pid1=$!

  local rc=0
  wait "$pid0" || rc=$?
  wait "$pid1" || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "error: $label child benchmark failed" >&2
    exit "$rc"
  fi
}

# Each child validates its parallel result against its own scalar oracle before
# timing. Because both children use the same seed/corpus, selected-page counts
# must also agree exactly within every placement mode.
run_pair local "--membind=0" "--membind=1"
run_pair interleave "--interleave=0,1" "--interleave=0,1"
run_pair remote "--membind=1" "--membind=0"

python3 - "$OUT" "$SHARD_PAGES" "$SIGNATURE_BITS" <<'PY'
import csv
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
shard_pages = int(sys.argv[2])
signature_bits = int(sys.argv[3])
total_pages = shard_pages * 2


def load(path: pathlib.Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise SystemExit(f"expected one benchmark row in {path}, got {len(rows)}")
    row = rows[0]
    return {
        "median_ns": int(row["median_ns"]),
        "selected_pages": int(row["selected_pages"]),
        "bits_compared": int(row["bits_compared"]),
    }

summary = []
for label in ("local", "interleave", "remote"):
    n0 = load(out / f"{label}-node0.csv")
    n1 = load(out / f"{label}-node1.csv")
    if n0["selected_pages"] != n1["selected_pages"]:
        raise SystemExit(
            f"{label}: mirrored candidate count mismatch: "
            f"{n0['selected_pages']} != {n1['selected_pages']}"
        )
    pair_ns = max(n0["median_ns"], n1["median_ns"])
    pages_per_second = total_pages / (pair_ns / 1e9)
    logical_bytes = total_pages * signature_bits / 8
    logical_gb_s = logical_bytes / (pair_ns / 1e9) / 1e9
    balance = min(n0["median_ns"], n1["median_ns"]) / pair_ns
    summary.append(
        (
            label,
            n0["median_ns"],
            n1["median_ns"],
            pair_ns,
            pages_per_second,
            logical_gb_s,
            balance,
            n0["selected_pages"],
        )
    )

with (out / "summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "placement",
            "node0_median_ns",
            "node1_median_ns",
            "pair_median_ns",
            "pages_per_second",
            "logical_gb_per_second",
            "socket_balance",
            "selected_pages_per_mirrored_shard",
        ]
    )
    for row in summary:
        w.writerow(
            [
                row[0],
                row[1],
                row[2],
                row[3],
                f"{row[4]:.6f}",
                f"{row[5]:.6f}",
                f"{row[6]:.6f}",
                row[7],
            ]
        )

print((out / "summary.csv").read_text(), end="")
PY
