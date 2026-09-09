# KVLab Coding Agent Instructions

You are working on **Memorithm/KVLab**, an experimental research bench for KV-state representation, manipulation, transfer, and causal analysis.

Your job is not merely to add code. Your job is to advance the research programme while preserving scientific validity, reproducibility, and clear ownership boundaries with the rest of the Memorithm ecosystem.

## Mission

Build KVLab into a reproducible experimental system that can:

1. capture exact KV states from supported model runtimes;
2. describe their geometry and provenance;
3. apply explicit, typed interventions;
4. inject the resulting state back into a model runtime;
5. continue decoding under controlled conditions;
6. compare transformed and reference executions at tensor, attention, logit, task, latency, and memory levels;
7. preserve enough metadata and artifacts to reproduce every reported result;
8. search for new manipulations only after the measurement and falsification substrate is trustworthy.

## Scientific anchor

The initial reference result is:

Taekyung Heo et al., **Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse**, arXiv:2608.03893v1, 4 Aug 2026.
https://arxiv.org/abs/2608.03893

Treat the paper as a baseline to reproduce, not as permission to generalize. Its experiments are primarily within-family, matched-KV, dense full-attention transfers. Mismatched KV geometry, cross-family transfer, hybrid attention, and attention-recurrent architectures remain research questions unless independently established by KVLab evidence.

## Non-negotiable scientific rules

- Never present an unexecuted idea as a result.
- Never present a benchmark improvement as a universal property.
- Never infer functional equivalence from MSE, cosine similarity, or R² alone.
- Never hide negative, equivalent, or inconclusive results.
- Never tune on a final holdout and then report that holdout as unbiased evidence.
- Never silently change a preregistered decision rule after observing the corresponding result.
- Distinguish clearly between `OBSERVATION`, `HYPOTHESIS`, `SUPPORTED`, `REFUTED`, `INCONCLUSIVE`, and `ENGINEERING_ONLY` claims.
- Store enough provenance to identify the exact models, revisions, tokenizer, runtime, prompt/corpus, geometry, dtype/numerical policy, seed, hardware/software environment, intervention, metrics, and code revision used.
- A cache transformation must be described explicitly enough to replay it.
- An experiment is incomplete if the reference execution has not been validated.

## Repository ownership boundaries

KVLab owns:

- KV state schemas and experiment-facing geometry;
- intervention semantics;
- experiment plans and preregistrations;
- capture/injection adapters;
- causal ablations and transformation experiments;
- evaluation contracts and evidence artifacts;
- KV-specific research reports.

Prefer reuse from other Memorithm repositories:

- **NNIS** owns native NVIDIA/CUDA runtime machinery and device-resident KV storage. Do not fork CUDA runtime infrastructure into KVLab when an adapter can call NNIS.
- **SciRust** owns reusable numerical/statistical/tensor primitives. Promote generic math there only after the KVLab implementation demonstrates a stable reusable contract.
- **TDI** owns its information/dynamics research programme. Do not move KVLab experiments into TDI unless the scientific question becomes explicitly a TDI line.
- **Forge** owns general automated algorithm/policy search. KVLab should expose a deterministic search surface rather than clone Forge.
- **ElasticXxx** owns adaptive resource-runtime policy. KVLab may measure cache migration/eviction; reusable elasticity policy belongs there.
- **ProofLab** owns formal mathematical proof workflows. Numerical evidence from KVLab is not proof.

## Development order

Follow `ROADMAP.md` in order unless repository evidence shows that an earlier dependency is missing.

The current priority is **KV-0**. Do not jump directly to semantic editing, automated search, or exotic transformations before capture -> intervention -> injection -> reference comparison works end to end.

## Required KV-0 implementation contracts

The first usable substrate should converge on the following conceptual interfaces. Names may evolve if Rust ergonomics require it, but responsibilities must stay separated.

```rust
KvStateDescriptor
KvGeometry
KvSide            // Key | Value
KvSelector        // layer/head/token/feature ranges
KvIntervention
KvTransformPlan
KvArtifactId
RunManifest
MetricRecord
ExperimentOutcome
```

A runtime adapter should eventually provide capabilities equivalent to:

```text
capture(model_session) -> KvArtifact
inject(model_session, KvArtifact) -> AppliedState
continue_decode(model_session, input) -> DecodeTrace
```

Do not make `kvlab-core` depend directly on CUDA, PyTorch, Transformers, or NNIS. Runtime-specific code belongs behind adapters.

## Intervention model

Prefer a composable typed representation rather than one-off experiment code. The long-term transformation vocabulary should be able to express operations such as:

```text
Select
Keep
Drop
Replace
Scale
AddNoise
Permute
Quantize
Project
LowRankApproximate
Merge
Splice
Move
Snapshot
Restore
Fork
CustomRecorded
```

Each transformation must declare:

- its target selector;
- required input geometry;
- output geometry;
- whether it is deterministic;
- parameters and parameter hashes;
- numerical policy;
- whether it mutates in place or creates a new artifact;
- validation conditions.

Do not implement the entire vocabulary during bootstrap. Add operators only with executable tests and a concrete experiment that uses them.

## Measurement hierarchy

When comparing a transformed cache with a reference cache, report multiple levels whenever applicable:

1. storage/geometry invariants;
2. tensor reconstruction diagnostics;
3. attention-output diagnostics;
4. logit/distribution divergence;
5. task behavior;
6. latency and memory cost;
7. stability across prompts, sequence lengths, turns, seeds, models, and hardware where the experiment requires it.

The NVIDIA baseline is especially important here: its reported mechanism analysis shows that raw reconstruction quality can be a poor predictor of downstream retention, while attention-output similarity is more informative across the tested pairs. KVLab must therefore keep functional metrics separate from reconstruction metrics.

## Reproducibility and artifacts

Machine-readable run records should be preferred over prose-only reports.

Every experiment run should eventually emit a manifest containing at least:

```text
schema_version
experiment_id
run_id
git_commit
model_source
model_target (optional)
model revisions
tokenizer
runtime/backend
hardware
software versions
input/corpus digest
seed policy
KV geometry
transform plan digest
metric definitions
raw metric values
artifact references
outcome classification
```

Large model weights and large raw KV tensors must not be committed to Git. Store manifests, hashes, compact fixtures, and instructions for obtaining/recreating external artifacts.

## Tests

Every new core type or transformation must have deterministic unit tests.

Every runtime adapter must have:

- shape/geometry validation tests;
- round-trip tests where supported;
- invalid-state rejection tests;
- at least one reference-oracle comparison;
- explicit skip/fail policy when required hardware is unavailable.

Never report a GPU experiment as passed if it was silently skipped.

## Rust quality gate

The default repository gate is:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
git diff --check
```

GPU-backed integration suites may add stricter environment-specific commands later.

## Pull-request discipline

Keep PRs scientifically and mechanically reviewable.

For each PR:

1. state the exact research/engineering capability added;
2. list invariants and non-claims;
3. include tests and the commands actually run;
4. include result artifacts only if the experiment was actually executed;
5. do not combine unrelated architectural rewrites with an experiment increment;
6. fix CI before merge;
7. after merge, reassess the next highest-leverage unfinished roadmap dependency.

## Cross-project improvement rule

When KVLab produces a genuinely reusable primitive that would improve another Memorithm project, identify the correct owner repository and port the generalized capability there instead of creating a KVLab-specific duplicate. Likewise, prefer importing stable capabilities from sibling repositories through narrow adapters.

## Stop conditions

Stop and document rather than guessing when:

- a required model/runtime artifact cannot be identified exactly;
- a paper detail needed for faithful reproduction is unspecified;
- an experiment would require changing the decision rule after looking at held-out evidence;
- hardware/runtime limitations make the claimed comparison invalid;
- a result cannot be distinguished from numerical or implementation error.

In those cases, produce a blocker record and the smallest experiment needed to resolve it.
