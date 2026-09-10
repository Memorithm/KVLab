# KV-C6 Stage 0 Preregistration — Causal Anisotropy after Geometric Isotropization

Status: **Stage 0 only. No confirmatory execution is authorized by this document.**

## Research question

After an orthogonal transform makes coordinate energy more isotropic, does functional/causal importance remain materially anisotropic under a fixed representation budget?

## Conjecture

Orthogonal rotation can reduce coordinate-energy anisotropy without equivalently reducing causal anisotropy. Under matched storage and intervention budgets, sensitivity- or recoverability-informed allocation may therefore preserve downstream behavior better than uniform or amplitude-only allocation.

## Null hypothesis

After energy isotropization and complete budget matching, causal/sensitivity-weighted allocation provides no reproducible held-out advantage over competent uniform and amplitude-based controls.

## Scientific boundary

This line is an experimental KVLab study. Orthogonal transforms or TurboQuant-like mechanisms are experimental mechanisms only; they are not evidence for the conjecture. No result from this line is automatically transferable to production policy, SciRust, FLAT-ATTENTION, NNIS, SLHAv2, TurboQuant, or ElasticXxx. Negative, equivalent, and inconclusive outcomes must be retained.

## Stage 0 deterministic micro-world

Use a deterministic vector-pair/attention-score micro-world before any real-model experiment.

For each fixture:

1. generate query/key/value blocks from an explicitly recorded seed;
2. compute an unmodified full-precision reference output;
3. apply a deterministic orthogonal transform `R` and verify the unquantized transformed system preserves the declared reference relation within the frozen numerical tolerance;
4. measure coordinate-energy anisotropy before and after rotation;
5. apply matched-budget perturbations or quantization;
6. perform single-coordinate and block interventions;
7. measure downstream output delta independently from coordinate magnitude;
8. record all inputs, seeds, transform identifiers, budgets and outputs for replay.

Identity transform and independent random orthogonal transforms are mandatory controls.

## Primary observables

The Stage 0 implementation must keep geometric and causal quantities separate.

- `energy_anisotropy_before`: frozen dispersion statistic over coordinate/block energy before rotation.
- `energy_anisotropy_after`: the same statistic after rotation.
- `intervention_delta`: downstream output delta caused by replacing, dropping or degrading one coordinate/block while all other state is held fixed.
- `sensitivity_per_byte`: intervention/sensitivity score divided by the exact bytes retained for the corresponding unit.
- `amplitude_per_byte`: magnitude-derived score divided by the same byte accounting.
- `recovery_error`: error after the preregistered reconstruction operation, when reconstruction is part of the fixture.

The concrete anisotropy statistic and aggregation rule must be frozen in code before any held-out comparison. Multiple metrics may be exploratory, but one primary metric must be designated before confirmatory use.

## Competent baselines

All allocation policies must receive exactly the same representational byte budget, including mandatory metadata.

1. uniform allocation;
2. amplitude-per-byte allocation;
3. immediate sensitivity-per-byte allocation using only information observable at the decision point;
4. random allocation with frozen seeds;
5. identity/no-rotation control;
6. rotation plus uniform allocation.

A future-aware recoverability oracle may be computed offline as an analysis bound, but future information must never be exposed to an online policy.

## Anti-leakage contract

For every online policy, construct paired traces with identical observable prefixes, identical metadata and identical decision-time state, but different future suffixes. The online decision must be identical across the pair. An intentionally leaking positive-control policy should fail this audit, proving that the audit can detect future leakage.

No hidden/held-out suffix may be used for feature construction, hyperparameter selection, policy ranking, stopping decisions or adaptive budget allocation.

## Budget accounting

The experimental record must account for:

- encoded K/V bytes;
- quantization scales/zero points or equivalent codec metadata;
- indices and block/page metadata required to reconstruct layout;
- optional reconstruction metadata;
- any additional retained state needed by a policy at inference time.

Compute/latency and transfer costs are secondary Stage 0 observables and must not be silently converted into byte savings.

## Falsifiable decision rules

Stage 0 is considered technically successful when the harness can reproduce the same fixture bit-for-bit where the arithmetic path is deterministic, or within the frozen numerical tolerance otherwise, and when it can distinguish energy anisotropy from intervention anisotropy.

The conjecture is **not** supported merely because rotation lowers the energy-anisotropy statistic. A later held-out study would require all of the following:

- the preregistered energy-isotropization condition is met;
- causal/intervention anisotropy remains measurably non-uniform;
- a causal/sensitivity policy improves the preregistered quality endpoint against uniform and amplitude baselines at the same complete budget;
- the effect reproduces across frozen seeds/families and survives the preregistered uncertainty criterion.

Failure of any required condition yields null-compatible, negative or inconclusive status rather than a positive claim.

## Provenance requirements

Every run record must include:

- Git commit SHA;
- fixture/schema version;
- seed set;
- transform identifier and parameters;
- policy identifier;
- exact byte budget and accounting breakdown;
- metric definitions/version;
- execution environment relevant to numerical reproducibility.

## Ecosystem responsibility

- KVLab owns the experiment, replay, interventions and scientific conclusion.
- FLAT-ATTENTION may supply portable attention semantics/kernels, but not the C6 conclusion.
- NNIS may supply NVIDIA/CUDA physical execution and movement, but not policy semantics.
- SLHAv2/TurboQuant may supply representation/rotation/codec mechanisms as explicit adapters.
- ElasticXxx may consume a separately qualified policy or threshold; it is not the scientific authority.
- SciRust should receive only generic mathematical primitives that have independent value and qualification.

## Stop condition for Stage 0

Stop before any confirmatory claim once the deterministic micro-world, identity/random-rotation controls, matched-budget policies, anti-leakage audit, provenance record and replay tests are implemented and green. A separate authorization/preregistration is required before executing a protected confirmatory study.