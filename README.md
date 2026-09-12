# KVLab

Experimental laboratory for KV-state representation, manipulation, transfer and causal analysis.

The 223-byte README was lying by omission. The lab already has a Python package (`kvlab/`), tests, and preregistered KV studies. Do not start a second KV repository.

## What this repository is

- Execution surface for campaigns C1–C11 / C6 as preregistered in `docs/`
- Causal analysis of KV-cache mechanics used by FLAT / TurboQuant / ADA
- Scientific source of truth for the first-class **Boolean KV Cache** programme
- Evidence packs belong here, not in a new lab

## What this repository is not

- Not the FLAT kernel owner (`Memorithm/FLAT-ATTENTION`)
- Not the TurboQuant codec owner (`Memorithm/TurboQuant`)
- Not a training run farm
- Not a place to silently change a preregistration after seeing results

## Layout

```text
docs/     preregistrations, protocols and research roadmaps
kvlab/    library
tests/    executable checks
```

## Priority research programmes

- [Boolean KV Cache — first-class memory-tier roadmap](docs/BOOLEAN_KV_CACHE_ROADMAP.md)
- [KV Tiering + Replay Study — Preregistration](docs/KV-TIERING-REPLAY-PREREGISTRATION.md)
- [Conjecture Research Programme](docs/CONJECTURE-RESEARCH-PROGRAMME.md)

### Boolean KV Cache

KVLab now distinguishes three architectures that must never be conflated:

1. **NKV** — numerical KV baseline;
2. **BIKV** — Boolean-indexed numerical KV, where Boolean state selects pages/blocks but exact numerical K/V remains authoritative;
3. **NBKV** — native Boolean KV for explicitly Boolean operations, research-only until separately qualified.

The dedicated roadmap covers packed Boolean page indexes, CPU/SIMD/multicore execution, dual-socket NUMA experiments, large-RAM qualification, portable GPU search, CPU+GPU cooperative execution, first-token readiness, and composition with the existing KVLab programme.

The first dedicated large-memory target is the user-reported dual-socket Dell T430-class server with approximately 125 GiB RAM and two CPUs reported as 32 cores each. Every run must detect and record the exact CPU SKU, physical/logical core count, NUMA topology, cache hierarchy and available instruction set rather than infer them from the chassis name.

## Canon

See `Memorithm/scirust-hub` `CATALOG.md` and ADR-0020. Existing preregistrations remain authoritative for their campaigns. New Boolean KV experiments must be preregistered under the dedicated roadmap before confirmatory runs; the new programme extends KVLab rather than redesigning or duplicating it.
