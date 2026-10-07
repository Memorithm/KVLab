#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

manifest="rust/bkv_scan/Cargo.toml"
lockfile="rust/bkv_scan/Cargo.lock"
evidence_dir="${EWW_EVIDENCE_DIR:-target/eww-evidence}"
rm -rf "$evidence_dir"
mkdir -p "$evidence_dir"
test -s "$lockfile"
cp "$lockfile" "$evidence_dir/Cargo.lock"
cargo metadata --locked --manifest-path "$manifest" --format-version 1 > "$evidence_dir/cargo-metadata.json"
sha256sum "$evidence_dir/Cargo.lock" "$evidence_dir/cargo-metadata.json" > "$evidence_dir/SHA256SUMS"

flat_executed_revision="$(
  sed -n 's/^flat-attention = {.*rev = "\([[:xdigit:]]\{40\}\)".*$/\1/p' "$manifest"
)"
test "${#flat_executed_revision}" -eq 40
grep -Fq \
  "source = \"git+https://github.com/Memorithm/FLAT-ATTENTION.git?rev=${flat_executed_revision}#${flat_executed_revision}\"" \
  "$lockfile"

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
  printf 'flat_attention_executed_revision=%s\n' "$flat_executed_revision"
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
  cargo run --quiet --release --locked --manifest-path "$manifest" "$@" | tee "$evidence_dir/$output"
  test -s "$evidence_dir/$output"
}

verify_flat_provenance() {
  local output="$1"
  local expected_schema="$2"
  local expected_contract_origin="$3"
  local header schema contract_origin executed_dependency remainder

  header="$(sed -n '1p' "$evidence_dir/$output")"
  test "$header" = "schema,contract_origin_revision,executed_dependency_revision${header#schema,contract_origin_revision,executed_dependency_revision}"
  IFS=, read -r schema contract_origin executed_dependency remainder < <(
    sed -n '2p' "$evidence_dir/$output"
  )
  test "$schema" = "$expected_schema"
  test "$contract_origin" = "$expected_contract_origin"
  test "$executed_dependency" = "$flat_executed_revision"
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
verify_flat_provenance \
  "eww-k4-flat-production-w64.csv" \
  "kvlab.eww-k4-flat-production-w64/v2" \
  "ae61403a4b4b4ed93ea8fa1bb4cd632be9038571"

run_case "EWW-K4 FLAT production W64 scale sweep" "eww-k4-flat-scale-sweep.csv" \
  --bin eww_flat_scale_sweep

run_case "EWW-K4b FLAT W64-to-W32 projection" "eww-k4b-flat-w32-projection.csv" \
  --bin eww_flat_w32_projection_bench
verify_flat_provenance \
  "eww-k4b-flat-w32-projection.csv" \
  "kvlab.eww-k4b-flat-w64-to-w32/v2" \
  "7bd9f1a2254d10329861e41a11694e7b4cffdd6b"

run_case "EWW-K4c FLAT packed-u16 shadow" "eww-k4c-flat-u16-shadow.csv" \
  --bin eww_flat_u16_shadow_bench
verify_flat_provenance \
  "eww-k4c-flat-u16-shadow.csv" \
  "kvlab.eww-k4c-packed-u16-shadow/v2" \
  "512224212ad834f19c407878ff4e6569fdcbe508"

run_case "EWW-K4c FLAT W32 uniform utilization" "eww-k4c-flat-w32-uniform-utilization.csv" \
  --bin eww_flat_w32_uniform_utilization
verify_flat_provenance \
  "eww-k4c-flat-w32-uniform-utilization.csv" \
  "kvlab.eww-k4c-w32-uniform-utilization/v2" \
  "5c49c1a88198c58390826f8cc1a6539ea52874a4"

run_case "EWW-K4g FLAT packed-u16 shader contract" "eww-k4g-flat-u16-shader-contract.csv" \
  --bin eww_flat_u16_shader_contract
verify_flat_provenance \
  "eww-k4g-flat-u16-shader-contract.csv" \
  "kvlab.eww-k4g-u16-shader-contract/v2" \
  "dff65361cb39b91e88a6d2bc99fc5469187809f4"

run_case "EWW-K4d FLAT packed-u16 eligibility" "eww-k4d-flat-u16-eligibility.csv" \
  --bin eww_flat_u16_eligibility

run_case "EWW-K4e FLAT packed-u16 uniform utilization" "eww-k4e-flat-u16-uniform-utilization.csv" \
  --bin eww_flat_u16_uniform_utilization

printf '%s\n' \
  "manifest.txt" \
  "Cargo.lock" \
  "cargo-metadata.json" \
  "SHA256SUMS" \
  "eww-k1-fixed-width.csv" \
  "eww-k2-transition-cost.csv" \
  "eww-k3-adaptive-policy.csv" \
  "eww-k4-flat-production-w64.csv" \
  "eww-k4-flat-scale-sweep.csv" \
  "eww-k4b-flat-w32-projection.csv" \
  "eww-k4c-flat-u16-shadow.csv" \
  "eww-k4c-flat-w32-uniform-utilization.csv" \
  "eww-k4g-flat-u16-shader-contract.csv" \
  > "$evidence_dir/files.txt"

while IFS= read -r relative; do
  test -s "$evidence_dir/$relative"
done < "$evidence_dir/files.txt"

echo
echo "EWW structural smoke complete."
echo "Evidence directory: $evidence_dir"
echo "These timings are runner-local smoke evidence only, not cross-hardware performance claims."
