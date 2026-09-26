# Elastic word-width sweep — preregistration v1

Status: preregistered development protocol. No performance, memory-saving,
latency, bandwidth, TTFT, TPOT or model-quality result is implied.

## Question

Can a KV control/index representation whose physical width is allowed to move
among 64, 128, 256, 512, 1024 and 2048 bits outperform one permanently fixed
width for declared workloads after transition, metadata and execution costs are
included?

This programme treats width as **representation state**, not as six unrelated
descriptor object models.

The generic adaptation contract belongs to Memorithm/ElasticXxx. KVLab owns
only the experiment design, cache-level fixtures, measurements and negative
results.

## Frozen width set

The v1 candidate set is:

```text
W = {64, 128, 256, 512, 1024, 2048} bits
  = {1, 2, 4, 8, 16, 32} × u64 lanes
```

No candidate may be added, removed or retuned after viewing confirmatory
results. A later width set requires a new protocol version.

## Research arms

### EWW-K0 — structural correctness

For every frozen width:

- exact packed-lane length is checked;
- scalar Hamming/equality semantics are deterministic;
- parallel packed scanning must return exactly the scalar candidate set;
- one-bit perturbations must have exact distance 1;
- non-zero unused tail bits remain invalid where a non-lane-aligned fixture is
  used.

This gate has no timing claim.

### EWW-K1 — fixed-width CPU baselines

Measure each width as a fixed configuration with no adaptation.

Required metrics:

- logical words processed;
- physical lane words read/written;
- bytes represented;
- elapsed time with warmup/repetition protocol;
- words/s and bytes/s derived from measured work;
- allocation count and peak memory only if actually exposed;
- cache/branch counters only if collected from a declared measurement source.

A theoretical cache-line argument is not measurement evidence.

### EWW-K2 — transition cost

Measure each ordered admissible transition:

```text
64 ↔ 128 ↔ 256 ↔ 512 ↔ 1024 ↔ 2048
```

and direct jumps when the backend supports them.

Record separately:

- source bytes;
- target bytes;
- bytes copied;
- bytes initialized;
- bytes discarded or externally rematerialized;
- transition latency;
- validation latency;
- verification latency;
- rollback latency when exercised.

A narrowing transition that would lose authoritative information is rejected
unless a separately declared domain codec/reconstruction contract proves the
target admissible.

### EWW-K3 — adaptive policy versus static baselines

Only after EWW-K0..K2 are retained, compare an ElasticXxx policy against every
fixed-width arm on the same workload sequence.

The adaptive arm must include:

- observation cost;
- forecast/planning cost;
- validation cost;
- transition cost;
- verification cost;
- rollback cost;
- hysteresis/cooldown if used.

Primary systems comparison is end-to-end elapsed time under an explicit memory
budget. Memory footprint and transition count remain separate objectives; they
must not be collapsed into a fabricated scalar score.

### EWW-K4 — FLAT-ATTENTION consumer

After the generic contract is qualified, evaluate width-aware control/index
planes in the existing FLAT paged/Boolean-KV execution paths.

Required outcomes include correctness parity, control metadata bytes, actual
page-selection cost, numerical K/V bytes touched/avoided, TTFT, TPOT and
tokens/s where a representative model/backend is available.

### EWW-K5 — NNIS native backend

Qualify native NVIDIA realization separately. Record exact hardware, driver,
CUDA/NVRTC versions, launch geometry, device residency and transfer boundaries.
Do not infer GPU benefit from CPU results.

## Mandatory static baselines

Every adaptive comparison must include all six fixed widths:

1. W64
2. W128
3. W256
4. W512
5. W1024
6. W2048

An adaptive policy is not considered beneficial merely because it beats one
poor static width. The retained report must expose every arm.

## Mandatory correctness invariants

- logical page/block identity is stable across a width transition;
- generation/epoch changes are explicit;
- stale descriptors fail closed;
- no physical block may be addressed by truncated or unvalidated metadata;
- packed comparison semantics are identical across scalar and parallel or
  accelerated paths;
- width adaptation cannot authorize a representation that the owning domain
  rejects;
- rollback restores an independently verified admissible state.

## Evidence boundary

KVLab may report only what its retained evidence measures.

In particular:

- logical bytes are not physical DRAM/HBM traffic;
- packed width is not cache residency;
- fewer lane words do not prove lower latency;
- a wider vector is not proof of one-instruction hardware execution;
- control-plane correctness is not model-quality evidence;
- synthetic fixtures are not representative-model performance evidence.

## Repository ownership

- **ElasticXxx** — generic elastic-width state, planning, validation,
  transaction/evidence lifecycle.
- **KVLab** — preregistration, falsification, cache/index experiments.
- **FLAT-ATTENTION** — attention/paged-KV consumption and portable GPU timing.
- **NNIS** — native NVIDIA execution qualification.
- **SLHAv2** — compressed-KV consumer semantics and quality/budget invariants.
- **SciRust** — reusable packed-bit/SIMD primitives only after independent
  consumer evidence supports promotion.
- **BooleanLab** — searched Boolean predicates/signatures; no runtime authority.

## Stop rule

If width adaptation does not improve a preregistered objective after its full
control and transition overhead is included, retain the negative result and use
the best admissible static configuration for that workload.
