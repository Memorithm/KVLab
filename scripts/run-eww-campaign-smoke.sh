#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

manifest="rust/bkv_scan/Cargo.toml"

echo "schema=kvlab.eww-campaign-smoke/v1"
echo "git_head=$(git rev-parse HEAD)"
echo "rustc_version=$(rustc --version)"
echo "cargo_version=$(cargo --version)"
echo "uname=$(uname -a)"

run_case() {
  local label="$1"
  shift
  echo
  echo "=== ${label} ==="
  cargo run --quiet --release --manifest-path "$manifest" "$@"
}

run_case "EWW-K1 fixed widths"   --bin eww_fixed_width_bench --   4096 1 3 42 5000

run_case "EWW-K2 transition costs"   --bin eww_transition_cost_bench --   2048 1 3 42

run_case "EWW-K3 adaptive policy"   --bin eww_adaptive_policy_bench --   2048 3 2 512 18446744073709551615 42

run_case "EWW-K4 FLAT production W64"   --bin eww_flat_production_w64_bench --   16 4096 32768 1 3

echo
echo "EWW structural smoke complete."
echo "These timings are runner-local smoke evidence only, not cross-hardware performance claims."
