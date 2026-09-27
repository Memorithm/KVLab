# EWW structural campaign smoke

The workflow `.github/workflows/eww-structural-smoke.yml` executes the retained
Elastic Word Width campaign harnesses in order:

1. EWW-K1 — six fixed-width baselines;
2. EWW-K2 — structural width-transition cost;
3. EWW-K3 — adaptive width policy versus static baselines;
4. EWW-K4 — FLAT production W64 page-map host qualification.

The runner records its Git commit, Rust compiler, Cargo version and host identity
before executing the campaign.

## Evidence boundary

The smoke exists to prove that the retained harnesses execute together against
the current pinned dependencies. Its elapsed times are **runner-local smoke
evidence only**.

Do not use this workflow to claim:

- cross-hardware speedup;
- physical DRAM/HBM traffic;
- cache residency;
- GPU performance;
- TTFT, TPOT or tokens/s;
- model quality;
- production benefit.

Those claims require their separately preregistered representative hardware,
model and measurement protocols.

The exact output remains retained in the GitHub Actions log for the workflow
run. A failed harness fails the workflow rather than silently omitting an arm.
