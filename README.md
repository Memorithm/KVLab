# KVLab

**Experimental Laboratory for KV-State Representation, Manipulation, Transfer, and Causal Analysis**

KVLab is a research bench for treating the transformer key/value cache as an experimentally manipulable intermediate state rather than only as a serving optimization.

The initial scientific anchor is:

- Taekyung Heo et al., **Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse**, arXiv:2608.03893v1, 4 Aug 2026: https://arxiv.org/abs/2608.03893

That work establishes a reproducible starting point: for several **matched-KV, within-family, dense full-attention** model pairs, useful cross-model structure can be captured with per-head closed-form mappings, cross-layer source selection, and RoPE factoring. KVLab must not silently generalize those findings beyond the conditions actually tested in the paper.

## Research objective

KVLab asks:

- what information is represented in a KV cache;
- where that information lives across layers, heads, tokens, K/V sides, and feature subspaces;
- how the representation evolves through a conversation;
- which parts are functionally necessary for a specified behavior;
- which transformations preserve, degrade, or deliberately alter downstream behavior;
- whether useful representations transfer across model sizes, families, KV geometries, and attention architectures;
- how cache manipulation interacts with latency, memory, robustness, isolation, and reproducibility.

The long-term experimental loop is:

```text
CAPTURE
-> CHARACTERIZE
-> INTERVENE
-> INJECT
-> CONTINUE DECODE
-> MEASURE
-> FALSIFY / REPLICATE
-> GENERALIZE
```

## Research programme

| Series | Question |
| --- | --- |
| **KV-0** | Can we capture, inject, replay, and reproduce the NVIDIA cross-model transfer baseline correctly? |
| **KV-1** | Which layer/head/token/K/V components causally matter for downstream behavior? |
| **KV-2** | What attention-visible subspaces are functionally important, and what is the effective KV dimension? |
| **KV-3** | Can transfer work when KV head counts or per-head dimensions do not match? |
| **KV-4** | Which KV relationships survive across model families? |
| **KV-5** | What is the minimum partial cache sufficient for a declared behavior? |
| **KV-6** | Which numerical and structural transformations preserve useful behavior? |
| **KV-7** | What can temporal cache surgery enable: snapshot, rewind, fork, rollback, splice, replay? |
| **KV-8** | Can controlled KV interventions produce localized semantic effects without uncontrolled degradation? |
| **KV-9** | Can automated search discover better transformations or cache policies under explicit constraints? |

See [`ROADMAP.md`](ROADMAP.md) for the staged implementation plan and [`AGENTS.md`](AGENTS.md) for coding-agent rules.

## Project boundaries

KVLab owns the **experimental semantics, intervention model, measurements, provenance, and evidence** for KV manipulation. It should reuse existing Memorithm projects rather than absorb their responsibilities:

- **NNIS** — native NVIDIA/CUDA execution backend and device-resident KV storage;
- **TDI** — information/dynamics analysis when a KV result becomes a TDI research question;
- **SciRust** — reusable numerical, statistical, tensor, and representation primitives;
- **Forge** — automated search over transformation policies;
- **ElasticXxx** — cache placement, migration, eviction, and resource-control policies when they become an elasticity problem;
- **ProofLab** — formalization only after an experimentally grounded claim becomes precise enough for proof.

## Evidence policy

Numerical similarity is not functional equivalence. Metrics such as MSE, cosine similarity, or R² are diagnostics, not proof that a transformed cache is behaviorally equivalent to the reference cache.

Every scientific result must record enough information to reproduce the run, including model identifiers/revisions, tokenizer identity, prompt or corpus identity, cache geometry, intervention, numerical policy, seeds where applicable, hardware/software environment, metrics, and raw result artifacts.

Positive, negative, equivalent, and inconclusive outcomes must all remain publishable artifacts of the bench.

## Bootstrap architecture

```text
KVLab/
  crates/
    kvlab-core/        typed experiment/intervention contracts
  docs/
    SCIENTIFIC_CONTRACT.md
    KV-0-BASELINE.md
  experiments/         staged experiment specifications and outputs
  adapters/            future NNIS / model-runtime adapters
  results/             machine-readable run artifacts (large payloads may live externally)
  ROADMAP.md
  AGENTS.md
```

Do not create crates merely to fill the diagram. Add a component only when it has an executable responsibility, tests, and a clear owner.

## Current status

KVLab is at **KV-0 bootstrap**. The repository does not yet claim a successful reproduction, a novel KV algorithm, cross-family transfer, effective-dimension result, or semantic intervention capability.
