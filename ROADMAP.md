# KVLab Roadmap — Massive Adaptive-KV Campaign for SLHAv2

Status: active, 2026-09-26.

## Mission

Find reproducible KV representation/routing policies that improve SLHAv2's
real-model quality/work/memory frontier. KVLab searches and falsifies; it does
not promote SLHA runtime behavior.

## Primary hypotheses

H1. Preserving high-impact attention structure can recover more model quality
per retained byte than uniformly increasing precision.

H2. Retained softmax mass is a stronger mechanistic quality signal than top-k
count alone for sparse/routed KV.

H3. Boolean/signature admission can cheaply eliminate numerical KV reads if its
false negatives are controlled by mass/quality rather than count alone.

H4. The optimal control/index width and representation state vary with workload;
an Elastic policy may outperform static states only after all transition and
control overhead is included.

H5. K and V, layers and heads have different sensitivity and should not be
forced into one uniform precision/residency policy without evidence.

## Campaign series

### SKV-0 — evaluator freeze
Status: implementation started.

Freeze canonical metrics:

- top-k overlap/recall;
- reference/candidate boundary-pair accuracy;
- retained and omitted softmax mass;
- mass-weighted misses;
- candidate density;
- deterministic tie rules.

The first implementation lives in `kvlab/slha_quality.py`.

### SKV-1 — synthetic falsification panel

Generate controlled score distributions:

- flat;
- one dominant key;
- multiple near-ties around the top-k boundary;
- heavy-tail;
- variable norms;
- adversarial sign collisions;
- recency-correlated and recency-independent relevance.

Compare exact top-k, Hamming/signature, norm, positional, random
matched-density and hybrid policies.

Purpose: falsify weak metrics cheaply, not claim model quality.

### CPS-2 — FLAT compact-preselection comparative quality bridge
Status: active development protocol.

Consume the exact FLAT CPS-1 producer at merge `ad1634fc922f6223dd3a83ac84154a82b1a35562` and compare its compact coordinate-projected candidate sets against all-accept, full-dimensional top-k, recent-tail and deterministic matched-random controls. The protocol is frozen in `docs/CPS2_COMPACT_PRESELECTION_PROTOCOL.md`.

Development metrics are row-level top-k recall, retained/omitted softmax mass, O/LSE error, selected density, selector score components and numerical pairs executed. The first panel is synthetic and explicitly retains an omitted-dominant-coordinate negative control. It is not a speed, physical-traffic, real-model-quality or runtime-promotion result.

Next gate: freeze real-model/model-tokenizer-dataset identities, coordinate/projection policy, budget, seeds, acceptance criteria and split identities before any protected population is observed.

### SKV-2 — real activation / real score capture

On frozen model/layer/head/query populations, retain exact reference scores as
evaluation-only truth and measure each candidate's ranking/mass frontier.

Required segmentation:

- per layer;
- per head;
- context position/age;
- query class when available;
- K versus V sensitivity;
- selected density/budget.

### SKV-3 — Boolean-KV mass-aware routing

Extend BIKV evidence with:

- retained-softmax-mass curves;
- mass-weighted false negatives;
- top-k boundary diagnostics;
- matched-density random and tail controls;
- first-token and steady-state separation.

No native Boolean KV promotion until the existing BKV-K9 gate is satisfied.

### SKV-4 — elastic signature/control widths

Execute the existing 64/128/256/512/1024/2048-bit preregistration with the
new quality metrics. Width is a representation state. Include transition,
validation, verification and rollback costs before judging an adaptive policy.

### SKV-5 — representation ladder

Compare combinations, not only isolated codecs:

- dense/full fidelity;
- mixed precision;
- SLHA compact state;
- residual-free state;
- Boolean-index-assisted sparse state;
- offload;
- bounded replay when semantics permit.

Keep K and V accounting separate.

### SKV-6 — query-aware versus query-agnostic importance

Evaluate:

- query-aware ranking/signature policies;
- reusable query-agnostic importance estimates;
- historical attention/frequency;
- age/position;
- reconstruction/context-preservation proxies.

Require multi-query evaluation before promoting a reusable importance score.

### SKV-7 — layer/head sensitivity and mixed policy

Build a sensitivity map and test static per-layer/head policies against a
single uniform policy. Any adaptive online policy must also beat the best
calibrated static policy after controller overhead.

### SKV-8 — FLAT physical consumer

Export frozen candidate sets/evidence into FLAT-ATTENTION. Measure:

- routing/front-end latency;
- numerical K/V bytes logically avoided;
- actual timing;
- TTFT/TPOT/tokens/s;
- O/LSE error;
- model quality where representative execution exists.

Logical savings are not physical traffic evidence.

### SKV-9 — NNIS / Thor native qualification

After portable/FLAT evidence freezes one candidate:

- device-resident Boolean/index state;
- native packed scans;
- survivor attention;
- CUDA-event timing;
- real allocation/transfer accounting.

### SKV-10 — adaptive repair

Test monotonic widening tiers. A deployable confidence signal may add survivors
when uncertainty is high. Exact dense softmax mass remains evaluation-only.

### SKV-11 — search engines

Only after SKV-0 and the budget are frozen:

- BooleanLab searches compact predicates;
- Forge searches Pareto policy/representation candidates;
- ADA searches attention mechanisms;
- TDI studies controller dynamics.

Every searched candidate is reevaluated by KVLab under the frozen protocol.

### SKV-12 — SLHAv2 export gate

For a candidate to be exported as a serious SLHAv2 development input, retain:

- exact producer revision;
- experiment identity;
- model/tokenizer/data identity if real-model;
- candidate parameters;
- all mandatory controls;
- ranking + mass metrics;
- quality metric where applicable;
- logical byte accounting;
- physical evidence for physical claims;
- canonical machine-readable bundle.

The SLHAv2 destination may still reject it.

## Parallelism

The campaign is deliberately parallel:

- Track A: ranking/mass metrics and real-score datasets;
- Track B: Boolean signatures and matched controls;
- Track C: elastic widths and transition costs;
- Track D: representation/tiering/replay;
- Track E: FLAT physical routing;
- Track F: NNIS Thor native path;
- Track G: policy search after evaluator freeze.

No track may use another track's confirmatory holdout as tuning data.

## Success condition

A useful result is not necessarily a positive speedup. The campaign succeeds
scientifically if it narrows the admissible design space with reproducible
positive **or negative** evidence. Product promotion requires a candidate that
passes SLHAv2's destination quality gate and then physical qualification.
