# KVLab Research and Engineering Roadmap

This roadmap is written for an autonomous coding agent. Follow the phases in dependency order. Do not skip measurement, provenance, or reference-oracle work to reach later research lines faster.

## Phase 0 — Bootstrap the trusted experiment substrate

Goal: make KV states first-class, typed, reproducible artifacts before implementing advanced manipulation.

Deliverables:

- Rust workspace with a small `kvlab-core` crate;
- typed KV geometry and selector contracts;
- deterministic transform-plan representation;
- run manifest and evidence classification types;
- canonical serialization suitable for hashing;
- unit tests for invalid geometry, selector bounds, deterministic identities, and manifest round-trips;
- CI enforcing formatting, check, clippy, tests, and diff hygiene.

Definition of done:

- a synthetic KV artifact can be described, selected, transformed by at least one identity/no-op operation, serialized, hashed, restored, and compared deterministically;
- no GPU/runtime dependency exists in `kvlab-core`;
- all validation gates pass.

Do not claim any model-level KV result in Phase 0.

---

## KV-0 — Observability, injection, and NVIDIA baseline reproduction

Scientific question:

> Can KVLab faithfully capture, transform, inject, and evaluate KV states, and can it reproduce the key qualitative findings of Heo et al. (arXiv:2608.03893v1) under documented conditions?

### KV-0.1 — Runtime adapter contract

Implement a runtime-neutral adapter API around:

```text
capture -> KvArtifact
inject -> AppliedKvState
continue_decode -> DecodeTrace
```

Requirements:

- exact geometry validation;
- explicit dtype and positional-encoding metadata;
- no silent tensor reshaping;
- no implicit head/layer remapping;
- model and tokenizer identity recorded in the run manifest.

Preferred first backend: **NNIS**, because Memorithm/NNIS already owns native CUDA execution and a device-resident KV cache. Add missing generic NNIS capabilities there if they belong to the runtime rather than duplicating them in KVLab.

Definition of done:

- capture/reference replay is behaviorally identical within declared numerical tolerance for a deterministic fixture;
- corrupted geometry is rejected;
- required GPU tests fail rather than silently pass when GPU execution is mandatory.

### KV-0.2 — Reference metrics

Implement a metric stack that separates:

- tensor reconstruction metrics;
- attention-output metrics where observable;
- logit/distribution metrics;
- downstream task metrics;
- latency and memory metrics.

Minimum diagnostics:

```text
MSE
relative L2
cosine similarity
R² where statistically meaningful
logit cosine / divergence
next-token agreement
```

Do not collapse these into one score.

### KV-0.3 — Reproduce the closed-form mapper

Reproduce the paper's core method before extending it:

- per-target-head linear/ridge mapping;
- independent K and V fits;
- cross-layer source selection;
- RoPE stripping for keys before mapping and target RoPE reapplication;
- configurable ridge regularization;
- calibration data manifest and digest;
- per-layer/per-head reconstruction diagnostics;
- downstream functional evaluation.

Paper-specific reference settings that should be reproduced when feasible and legally/operationally available:

- ridge `lambda = 0.01`;
- calibration on 500 FineWeb-Edu sequences of length 1024;
- source-layer count sweep over `{1,2,4,6,8,10,12,16,20,24,all}` where the model depths permit it;
- matched-KV within-family pairs as the first reproduction target;
- report functional retention separately from reconstruction quality.

The repository must record deviations from the paper rather than silently replacing them.

### KV-0.4 — Mechanism check

Test the paper's qualitative mechanism claim:

> downstream quality depends strongly on where residual mapping error lands relative to attention-sensitive subspaces, not only on global reconstruction error.

Implement attention-aware diagnostics if the backend exposes the required tensors. At minimum, compare reconstruction metrics against functional retention rather than assuming monotonicity.

Definition of done for KV-0:

- capture/injection works end to end;
- a documented reproduction attempt exists with exact environment and model revisions;
- the result is classified as replicated, partially replicated, refuted under the tested conditions, or inconclusive;
- raw machine-readable evidence is retained;
- no extension result is mixed into the reproduction claim.

---

## KV-1 — Causal atlas of the KV cache

Scientific question:

> Which components of a cache are causally necessary for a declared downstream behavior?

Intervention axes:

- layer;
- KV head;
- token position/range;
- Key vs Value;
- feature dimension/range;
- combinations of the above.

Initial interventions:

```text
zero
replace-with-reference
replace-with-other-position
shuffle
scale
controlled noise
mask/drop where the runtime semantics permit it
```

Measurements:

- attention-output change;
- logit distribution change;
- next-token agreement;
- task accuracy/quality;
- stability across prompts.

Required controls:

- no-op transformation;
- intervention-size matched random controls;
- repeated prompts/seeds where stochasticity exists;
- reference-cache replay.

Output:

A machine-readable sensitivity atlas, not merely a heatmap image.

---

## KV-2 — Functional subspaces and effective KV dimension

Scientific question:

> Is the functionally important KV representation lower-dimensional than its nominal tensor dimension under specified tasks and query distributions?

Starting hypothesis motivated by the NVIDIA paper, not established by it:

- residual error aligned with attention-sensitive query subspaces may matter more than equal-magnitude error in weakly observed directions.

Experiments:

- compute query-side SVD/PCA or alternative declared basis where observable;
- decompose K perturbations into high-visibility and low-visibility directions;
- sweep retained rank / removed subspace dimension;
- compare equal-energy perturbations in different subspaces;
- repeat for V using appropriate downstream observability diagnostics rather than assuming K and V behave identically.

Do not claim a universal "effective KV dimension". Any result must be conditioned on model, layer/head, task distribution, context regime, and metric.

Promotion opportunity:

Reusable subspace/statistical primitives should move to SciRust once stable.

---

## KV-3 — Mismatched KV geometry

Scientific question:

> Can a useful cache mapping exist when source and target KV head counts and/or per-head dimensions differ?

This is explicitly outside the tested matched-KV scope of the initial paper.

Experiments:

- MHA -> GQA;
- GQA -> MHA;
- GQA head-count changes;
- head-dimension expansion/contraction;
- combinations with differing depth.

Candidate mapping families:

- block linear maps;
- head mixing matrices;
- learned low-rank maps;
- structured projections;
- constrained assignment/matching;
- nonlinear baselines only after linear controls.

Controls:

- parameter-count matched mapper comparisons;
- simple copy/average/repeat baselines;
- random orthogonal/projection baselines where dimensionally valid.

---

## KV-4 — Cross-family transfer

Scientific question:

> Which KV relationships, if any, survive across separately trained model families?

Examples may include Qwen, Llama, Mistral/Ministral, Gemma, or other openly available models, subject to exact tokenizer and architecture constraints.

Required decomposition of failure modes:

- tokenizer mismatch;
- positional-encoding mismatch;
- head geometry mismatch;
- depth mismatch;
- representation mismatch;
- post-training/alignment mismatch.

Do not label cross-family failure as proof that no transferable structure exists. Classify only the tested mapping family and conditions.

Longer-term question:

- does there exist a shared intermediate KV representation that reduces pairwise mapping cost?

---

## KV-5 — Partial transfer and minimum sufficient cache

Scientific question:

> How little of the original cache can be preserved or transferred while retaining a declared capability?

Axes:

- subset of source layers;
- subset of target layers receiving transferred state;
- subset of heads;
- subset of tokens;
- K-only / V-only / mixed budgets;
- subspace/rank budget.

Optimization target must be multi-objective and explicit:

```text
functional quality
memory bytes
mapping latency
decode latency
robustness
```

The output should be Pareto evidence, not a single hand-picked operating point.

---

## KV-6 — Numerical and structural transformations

Scientific question:

> Which transformations preserve useful function, and where do their failure boundaries lie?

Families:

- quantization;
- clipping/scaling;
- additive noise;
- low-rank approximation;
- sparsification;
- clustering/codebooks;
- token pooling/merging;
- head merging;
- structured permutation;
- entropy coding where representation permits;
- combinations of these operations.

Every transform must preserve a replayable `KvTransformPlan`.

Cross-project opportunities:

- generalized representation accounting may belong in SciRust;
- adaptive memory policies may belong in ElasticXxx;
- native CUDA kernels belong in NNIS.

---

## KV-7 — Temporal cache surgery

Scientific question:

> What stateful operations become possible when the cache is treated as a versioned execution state?

Operations:

- snapshot;
- restore;
- truncate;
- rewind;
- fork;
- copy-on-write branch;
- splice segments from compatible histories;
- rollback after speculative continuation;
- replay under an alternate continuation.

Measurements:

- exact/approximate replay fidelity;
- semantic drift;
- branch isolation;
- state-copy cost;
- memory amplification;
- failure on incompatible positional state.

Potential NNIS improvements should be implemented in NNIS when they are runtime primitives rather than research-specific semantics.

---

## KV-8 — Controlled semantic intervention

Scientific question:

> Can a localized cache transformation produce a predictable, bounded behavioral change while preserving unrelated capabilities?

Prerequisites:

- KV-1 causal atlas;
- validated injection/replay;
- strong unrelated-behavior controls;
- explicit target behavior and non-target behavior metrics.

Do not call an intervention "semantic editing" merely because one sampled output changes. Require replication across a declared evaluation set and report collateral effects.

Research directions:

- fact/context suppression;
- retrieval influence changes;
- controlled emphasis/de-emphasis;
- branch-specific state modification;
- state transfer between compatible contexts.

---

## KV-9 — Automated discovery

Scientific question:

> Can an execution-driven search system discover cache transforms or policies that dominate hand-designed baselines under explicit constraints?

Prerequisites:

- deterministic transform IR;
- trustworthy evaluator;
- resource accounting;
- train/validation/holdout discipline;
- safe search bounds.

Preferred integration: expose KVLab candidates/evaluators to **Forge** rather than embedding a second generic search engine.

Candidate search space:

```text
selector
-> transform sequence
-> transform parameters
-> resource policy
```

Objective vector:

```text
functional retention / task quality
memory footprint
transform latency
decode latency
robustness
```

Final holdouts must remain unavailable to ordinary search iterations.

---

## Cross-cutting engineering milestones

### Artifact format

Develop a versioned artifact schema for:

- geometry;
- model/runtime identity;
- transform plan;
- metrics;
- provenance;
- compact synthetic KV fixtures.

Avoid committing production-size raw caches to Git.

### Determinism

Classify each operation as deterministic, seed-deterministic, backend-deterministic, or nondeterministic. Record the class in evidence.

### Visualization

Only after machine-readable evidence exists, add tools to visualize:

- layer-to-layer mapping matrices;
- head/token sensitivity maps;
- subspace spectra;
- quality vs. memory/latency Pareto fronts;
- temporal branch trees.

Plots are views of evidence, not the evidence source of truth.

### Hardware backends

Start with one correct backend. Add others behind the same contract only when they enable a concrete research question.

Potential backends:

- NNIS/native CUDA;
- a Python/Hugging Face reference adapter for reproduction convenience if needed;
- other inference runtimes only when justified by experiments.

### Security and isolation

When shared/persistent caches are introduced, add explicit experiments for:

- stale state reuse;
- tenant/session isolation;
- cache poisoning/corruption tolerance;
- timing/metadata leakage;
- zeroization vs. logical reset.

Keep security experiments isolated and defensive; do not turn KVLab into an exploitation toolkit.

---

## Agent execution policy

At the start of each work cycle:

1. inspect the actual `main` branch, open PRs, CI, and experiment status;
2. identify the earliest unfinished dependency in this roadmap;
3. implement the smallest end-to-end slice that produces executable evidence;
4. run the full relevant validation gate;
5. open a focused PR;
6. fix CI and review findings;
7. merge only when green;
8. update roadmap/status docs when repository reality changes;
9. immediately identify reusable improvements for NNIS, SciRust, TDI, Forge, ElasticXxx, or ProofLab and port them to the correct owner repository when justified.

Prefer forward progress through verified increments over large speculative rewrites.

## Current next engineering slice

**Build Phase 0 / `kvlab-core` first.**

The first coding PR after this bootstrap should implement:

- `KvGeometry`;
- `KvSide`;
- validated ranges/selectors;
- `KvArtifactDescriptor`;
- a minimal deterministic `KvTransformPlan` with `Identity` plus one simple slice-local transform suitable for synthetic tests;
- canonical serialization/hash contract;
- `RunManifest` and `ExperimentOutcome`;
- deterministic unit tests;
- CI.

Stop before runtime-specific capture code if the core artifact/intervention contracts are not yet stable and tested.
