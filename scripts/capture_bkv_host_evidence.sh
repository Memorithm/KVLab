#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <output-directory>" >&2
  exit 2
fi

out=$1
mkdir -p "$out"

{
  echo "captured_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -srvm)"
  echo "arch=$(uname -m)"
  echo "git_commit=$(git rev-parse HEAD)"
  if [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    echo "cpu0_scaling_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  else
    echo "cpu0_scaling_governor=UNAVAILABLE"
  fi
  if [[ -r /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    echo "intel_pstate_no_turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
  else
    echo "intel_pstate_no_turbo=UNAVAILABLE"
  fi
} >"$out/host.txt"

lscpu >"$out/lscpu.txt"
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE >"$out/lscpu-topology.txt"
cat /proc/cpuinfo >"$out/proc-cpuinfo.txt"
cat /proc/meminfo >"$out/proc-meminfo.txt"

if command -v numactl >/dev/null 2>&1; then
  numactl --hardware >"$out/numactl-hardware.txt"
else
  printf '%s\n' 'UNAVAILABLE: numactl not installed' >"$out/numactl-hardware.txt"
fi

if command -v git >/dev/null 2>&1; then
  git status --short >"$out/git-status.txt"
fi

printf 'evidence_pack=%s\n' "$out"
