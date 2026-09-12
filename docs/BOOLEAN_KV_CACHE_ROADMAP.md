# Boolean KV Cache Research Roadmap

Status: priority KVLab programme co-developed with FLAT-ATTENTION and BooleanLab.

## Mission

KVLab will treat the Boolean KV Cache as a first-class experimental memory tier rather than as a temporary mask. The initial scientific question is whether a compact Boolean representation of KV state can reduce numerical KV reads and attention work enough to improve end-to-end inference while preserving a declared quality/correctness target.

The programme distinguishes three architectures:

1. **NKV** — numerical KV baseline;
2. **BIKV** — Boolean-indexed numerical KV: Boolean state selects pages/blocks, exact numerical K/V remains authoritative;
3. **NBKV** — native Boolean KV: only for operations whose semantics are explicitly Boolean and independently qualified.

No result may silently conflate these architectures.

## Primary hypothesis

For a preregistered workload and routing policy:

```text
T_boolean_index + T_numerical_survivors < T_numerical_baseline
```

subject to a declared quality/correctness constraint and with all Boolean metadata, transfer and synchronization costs included.

A second memory-efficiency hypothesis is:

```text
numerical_KV_bytes_avoided / Boolean_KV_bytes_read >> 1
```

This ratio must be measured, not assumed.

## First-class cache object

Each Boolean KV page should expose a versioned experimental representation equivalent to:

```text
BooleanKvPage
  page_id
  token_range
  valid_count
  packed_key_signature
  optional packed_value_signature
  visibility / retention / routing bits
  optional preregistered position-age-frequency metadata
  generation / reset epoch
```

The exact bit width, packing order, page aggregation rule and tail semantics are part of the experiment manifest.

## Experimental backends

Boolean KV must be evaluated on multiple execution organs, not only GPU:

- scalar Rust/Python-compatible oracle as applicable;
- packed CPU baseline;
- CPU SIMD where available;
- CPU multicore;
- CPU dual-socket / NUMA;
- portable GPU/WGPU through FLAT-ATTENTION;
- optional native GPU bit-matrix qualification through NNIS;
- cooperative CPU + GPU execution.

CPU is a first-class backend because small/irregular Boolean decisions, very large host-resident indexes and low launch overhead may change the optimum.

## Dell dual-socket qualification target

A dedicated campaign will target the user-reported Dell T430-class server with approximately 125 GiB RAM and two CPUs reported as 32 cores each. The exact CPU SKU, physical/logical core counts, cache hierarchy, NUMA layout and ISA capabilities must be captured at runtime in every evidence pack and must not be inferred from the chassis name.

Required topology capture:

```text
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
numactl --hardware
/proc/cpuinfo feature summary
```

The campaign must compare:

- one core;
- one socket;
- both sockets;
- local NUMA memory;
- remote NUMA memory;
- interleaved memory;
- page sharding by socket;
- replicated read-mostly signatures;
- replication of tiny query signatures to both sockets;
- candidate-set merge cost.

No scaling claim is valid without physical-core and NUMA accounting.

## Boolean Matrix Equation search

The Boolean KV index must support baseline Boolean matrix families before searched equations:

```text
C_ij = R_k(Phi(A_ik, B_kj))
```

Baseline families:

- OR-AND Boolean product;
- XOR-AND / GF(2);
- XNOR + popcount;
- thresholded popcount;
- simple preregistered block/page predicates.

BooleanLab owns discovery/equivalence screening of more complex `Phi` and `R`. KVLab owns cache-level evaluation and falsification.

## Campaigns

### BKV-K0 — Specification and manifest

Create:

- versioned Boolean KV schema;
- exact bit accounting;
- experiment manifest fields;
- lifecycle semantics for append/reset/reuse;
- numerical KV oracle mapping.

Gate: deterministic encoding and stale-page failure tests.

### BKV-K1 — Signature and page aggregation controls

Compare:

- random signatures;
- sign/binary projections;
- positional/structural controls;
- XNOR-popcount signatures;
- any later BooleanLab candidate only after preregistration.

Measure recall, false negatives, metadata size and candidate density.

### BKV-K2 — CPU packed baseline

Implement/benchmark packed bit operations and page scans. Establish one-core reference and scaling protocol.

Metrics:

- pages/s;
- bits compared/s;
- queries/s;
- latency;
- effective memory bandwidth;
- cycles/page where measurable.

### BKV-K3 — CPU multicore scaling

Sweep physical cores from 1 to all available cores.

Report:

```text
S(p) = T(1) / T(p)
E(p) = S(p) / p
```

Retain negative scaling regimes.

### BKV-K4 — NUMA / large-memory campaign

On the dual-socket server, evaluate local/remote/interleaved/replicated/sharded layouts and query-signature replication.

Primary outputs:

- local vs remote latency/bandwidth;
- merge overhead;
- best page ownership strategy;
- Boolean index capacity actually usable under the measured RAM budget.

Do not equate nominal RAM bits with usable cache entries without accounting for metadata, allocator overhead and operating-system headroom.

### BKV-K5 — Boolean-indexed numerical KV

Integrate with FLAT-ATTENTION page/block routing.

Compare:

- full numerical KV;
- paged numerical KV;
- random matched-density routing;
- structural/positional routing;
- Boolean routing + exact numerical FLAT.

Required metrics include first-token latency, TPOT, tokens/s, candidate density, quality, O/LSE error, Boolean overhead and numerical KV bytes avoided.

### BKV-K6 — CPU/GPU cooperative pipeline

Test CPU Boolean search in parallel with GPU numerical attention.

Qualification requires real trace/timing evidence of overlap and explicit accounting of query-signature transfer, candidate bitmap transfer, synchronization and backpressure.

### BKV-K7 — Portable GPU Boolean KV search

Through FLAT-ATTENTION, compare the same Boolean index semantics on WGPU.

The CPU and GPU paths must produce matching candidate sets for identical frozen inputs/policies before performance comparison.

### BKV-K8 — First-token readiness

Require Boolean KV metadata to be built during prefill so the first generated token can route through the Boolean tier immediately.

Measure first-token decode separately from steady state.

### BKV-K9 — Native Boolean KV gate

Only after BIKV qualification, test operations that can remain in the Boolean domain without numerical K/V access.

NBKV must have its own semantics and downstream quality metrics. It is not assumed equivalent to numerical KV.

### BKV-K10 — Composition with existing KVLab mechanisms

Ablate Boolean KV with:

- paged KV;
- prefix reuse;
- quantized KV;
- sliding window / eviction;
- sparse query-aware reads;
- tiering/offload;
- compatible GQA/MQA;
- cross-model transformation only where architecture/protocol permits.

Use factorial/ablation designs to separate interactions from additive gains.

### BKV-K11 — Adaptive placement gate

Only after stable evidence, expose CPU/GPU/NUMA placement and routing policy to ElasticXxx. The controller must retain dense fallback and rollback.

## Mandatory baselines

Each confirmatory campaign must include, where applicable:

1. full numerical KV/native attention;
2. paged numerical KV;
3. random matched-density selection;
4. simple structural/positional selection;
5. Boolean KV CPU;
6. Boolean KV GPU;
7. CPU+GPU cooperative;
8. native Boolean KV only for semantically valid tasks.

## Mandatory metrics

### Correctness / quality

- candidate recall against declared dense target;
- false-negative rate;
- O/LSE error for BIKV;
- downstream loss/perplexity/task quality when available;
- reset/reuse correctness.

### Memory / traffic

- Boolean bits/token and bits/page;
- total Boolean index bytes;
- metadata overhead;
- numerical KV bytes touched;
- numerical KV bytes avoided;
- Boolean bytes read;
- host↔device transfer bytes;
- local/remote NUMA traffic when measurable;
- fragmentation and allocator overhead.

### Performance

- Boolean search latency;
- first-token latency;
- TPOT;
- tokens/s;
- pages/s;
- bits compared/s;
- effective bandwidth;
- scaling efficiency;
- synchronization and dispatch count.

### Energy/economics

Only report when measured inputs are known:

- joules/query or joules/token;
- throughput/watt;
- throughput per known hardware acquisition cost.

Never use guessed purchase prices.

## Experiment manifest additions

Every Boolean KV run must record:

- repository and exact commit SHAs for KVLab/FLAT/BooleanLab/SciRust as applicable;
- machine identity and topology;
- physical/logical cores and sockets;
- NUMA nodes;
- RAM available to the process;
- ISA/capability flags actually detected;
- page size;
- signature width;
- Boolean equation/policy identifier;
- candidate density;
- numerical KV precision/layout;
- context length;
- batch/concurrency;
- warmup/repetitions/seeds;
- calibration/validation/holdout split;
- all measured metrics and negative results.

## Ownership

- **KVLab**: experiment design, cache semantics, manifests, statistical evidence, negative results.
- **BooleanLab**: Boolean function/BME discovery and equivalence.
- **FLAT-ATTENTION**: page/block consumer, exact attention, portable GPU and heterogeneous scheduling.
- **SciRust**: promoted reusable packed-bit/SIMD/matrix/statistical primitives.
- **NNIS**: optional native hardware qualification.
- **ElasticXxx**: adaptive placement only after qualification.

## Scientific stop rule

A Boolean KV candidate is promoted only when it passes its preregistered quality/correctness gate and improves at least one declared systems objective after all Boolean overhead is included. If it saves arithmetic but loses end-to-end latency or quality, record the result as negative and keep the numerical fallback.
