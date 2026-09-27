#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

manifest="rust/bkv_scan/Cargo.toml"
evidence_dir="${EWW_EVIDENCE_DIR:-target/eww-evidence}"
rm -rf "$evidence_dir"
mkdir -p "$evidence_dir"

git_head="$(git rev-parse HEAD)"
rustc_version="$(rustc --version)"
cargo_version="$(cargo --version)"
uname_value="$(uname -a)"

{
  printf 'schema=kvlab.eww-campaign-smoke/v4\n'
  printf 'git_head=%s\n' "$git_head"
  printf 'rustc_version=%s\n' "$rustc_version"
  printf 'cargo_version=%s\n' "$cargo_version"
  printf 'uname=%s\n' "$uname_value"
  printf 'evidence_scope=runner-local-structural-smoke\n'
  printf 'cross_hardware_claim=false\n'
  printf 'gpu_performance_claim=false\n'
  printf 'model_quality_claim=false\n'
} | tee "$evidence_dir/manifest.txt"

run_case() {
  local label="$1"
  local output="$2"
  shift 2
  echo
  echo "=== ${label} ==="
  cargo run --quiet --release --manifest-path "$manifest" "$@" | tee "$evidence_dir/$output"
  test -s "$evidence_dir/$output"
}

run_case "EWW-K1 fixed widths" "eww-k1-fixed-width.csv" \
  --bin eww_fixed_width_bench -- \
  4096 1 3 42 5000

run_case "EWW-K2 transition costs" "eww-k2-transition-cost.csv" \
  --bin eww_transition_cost_bench -- \
  2048 1 3 42

run_case "EWW-K3 adaptive policy" "eww-k3-adaptive-policy.csv" \
  --bin eww_adaptive_policy_bench -- \
  2048 3 2 512 18446744073709551615 42

run_case "EWW-K4 FLAT production W64" "eww-k4-flat-production-w64.csv" \
  --bin eww_flat_production_w64_bench -- \
  16 4096 32768 1 3

run_case "EWW-K4 FLAT production W64 scale sweep" "eww-k4-flat-scale-sweep.csv" \
  --bin eww_flat_scale_sweep

run_case "EWW-K4b FLAT W64 to W32 host projection" "eww-k4b-flat-w32-projection.csv" \
  --bin eww_flat_w32_projection_bench

printf '%s\n' \
  "manifest.txt" \
  "eww-k1-fixed-width.csv" \
  "eww-k2-transition-cost.csv" \
  "eww-k3-adaptive-policy.csv" \
  "eww-k4-flat-production-w64.csv" \
  "eww-k4-flat-scale-sweep.csv" \
  > "$evidence_dir/files.txt"

while IFS= read -r relative; do
  test -s "$evidence_dir/$relative"
done < "$evidence_dir/files.txt"

echo
echo "EWW structural smoke complete."
echo "Evidence directory: $evidence_dir"
echo "These timings are runner-local smoke evidence only, not cross-hardware performance claims."
