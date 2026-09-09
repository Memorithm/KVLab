# KVLab Conjecture Research Programme

Status: **Stage 0 bootstrap; no confirmatory execution authorised**.

KVLab owns the experimental questions below because they require direct observation, manipulation, transfer and replay of KV state. External projects may provide mechanisms, but KVLab remains the scientific bench and must compare them under matched budgets and frozen metrics.

## KV-C1 — Recoverable Information per Byte

### Conjecture
A memory allocation policy based on prospective recoverable information per byte can outperform age/LRU, magnitude and attention-score heuristics at equal memory cost.

### Null
After matched-budget controls, recoverability-based allocation provides no held-out advantage over competent baselines.

### Stage 0 requirements
- define full-cache oracle and replay semantics;
- define recoverability target and horizon without future-label leakage;
- freeze byte accounting including metadata and transfer cost;
- implement LRU/age, magnitude, attention and random baselines;
- define quality, recovery, latency and memory endpoints;
- preserve negative/equivalent results.

### First executable slice
A deterministic synthetic KV trace format with full-cache reference outputs and an evaluator that removes/quantizes one region at a time and records downstream output delta plus exact byte savings.

## KV-C6 — Causal Anisotropy after Geometric Isotropization

### Conjecture
Orthogonal rotation can isotropize coordinate energy while leaving functional/causal importance strongly anisotropic, implying that residual correction should follow functional importance rather than amplitude alone.

### Null
Once energy is isotropized, causal/sensitivity-weighted correction gives no advantage over uniform or amplitude-based correction at the same bit budget.

### Stage 0 requirements
- use deterministic orthogonal transforms with recorded seeds;
- separate energy isotropy metric from intervention effect metric;
- compare uniform, amplitude, sensitivity and recoverability/causal allocation;
- include random rotations and identity transform controls;
- use TurboQuant only as a mechanism/reference, not as authority for the result.

### First executable slice
A vector-pair/attention micro-world where rotation preserves exact unquantized scores, followed by controlled quantization and one-coordinate/one-block interventions to measure energy vs functional importance divergence.

## KV-C11 — Multi-Resolution KV Pyramid

### Conjecture
A HOT/WARM/COLD hierarchy with on-demand reconstruction can dominate a flat KV representation on a joint fidelity/memory/transfer frontier for long-context workloads.

### Null
Any benefit is explained by using more compute/transfer or by easier workloads; under complete accounting the hierarchy does not improve the frontier.

### Stage 0 requirements
- full-cache exact/reference tier;
- explicit HOT/WARM/COLD representation contracts;
- reconstruction/page-in semantics;
- byte, compute and transfer accounting;
- flat-cache, uniform-quantization and eviction baselines;
- replayable access/intervention traces;
- no production claim without real-model validation.

### First executable slice
Implement a deterministic tier transition trace and evaluator independent of SLHAv2/ElasticXxx internals, then add adapters to their public contracts rather than copying code.

## Ecosystem boundaries

- FLAT-ATTENTION owns attention semantics and portable attention kernels.
- NNIS owns native NVIDIA/CUDA placement and physical movement.
- SLHAv2/TurboQuant may provide codecs/tiering mechanisms.
- ElasticXxx may provide adaptive-resource contracts.
- CCOS may provide causal/replay concepts or adapters.
- TDI may consume qualified KVLab observables, but does not determine KVLab results.

## Evidence rule
No metric, benchmark, compression ratio or policy advantage becomes a scientific claim until the corresponding protocol is frozen, baselines are implemented, and the result is reproduced under the recorded environment.