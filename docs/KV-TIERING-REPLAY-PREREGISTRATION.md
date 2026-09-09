# KVLab KV Tiering + Replay Study — Preregistration

Status: NON-FINAL PREREGISTRATION ONLY

This document defines a falsifiable KV-state experiment. It does not claim that any compression, retention, eviction or replay policy improves quality or performance until the experiment is executed under the frozen protocol below.

## Ownership boundaries

- **KVLab** owns the experimental design, interventions, causal comparisons, provenance and analysis.
- **FLAT-ATTENTION** owns portable attention/KV semantics, selection/tiering policies and reusable checkpoint/replay contracts.
- **NNIS** owns NVIDIA/CUDA physical placement, memory movement and backend-specific kernels.

KVLab must consume versioned interfaces rather than copy their implementation.

## External prior art motivating the study

Recent mixed-format paged-KV work, including *Minima-KV: Retention-Preserving KV Cache Compression with Mixed-Format Paged Attention* (Kozyrev & Maiboroda, arXiv:2608.23834, 2026-08-24), motivates testing retention-preserving tiering against irreversible eviction under matched memory budgets. Its reported results are prior art only and are not evidence for Memorithm.

*QEvict: Recoverable Quantized KV Eviction for Attention-Drift-Robust Long-Context Decoding* (Garg et al., arXiv:2608.05326, 2026-08-05) motivates explicitly measuring whether previously low-importance state becomes important later. Its `Future Missed Mass` and `Global LIR` diagnostics are candidate reactivation metrics only; their exact Memorithm definitions and oracle requirements must be frozen on Development before use.

*Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning* (Wang et al., arXiv:2609.03430, 2026-09-03) reports that a prompt-protected random per-head eviction control can be competitive with scored eviction on the authors' tested reasoning workloads. KVLab therefore includes a matched random control so a complex selector is not credited merely for structural prompt protection. This external result is not evidence that random eviction will work on Memorithm workloads.

## Memorithm substrate freeze

Initial FLAT checkpoint/replay semantic reference:

- repository: `Memorithm/FLAT-ATTENTION`
- merged PR: `#186`
- merge commit: `e80dd987c2577a89456edb21d7910637ecc145ae`
- capability: lineage-safe metadata checkpoints/restores for paged resident KV; not a physical byte snapshot.

Before scored runs, replace any mutable repository references with exact commit/artifact identities for every implementation used.

## Research question

Under an identical live-KV memory budget and identical generation workload, does recoverable KV tiering preserve the quality/correctness frontier better than irreversible eviction, and when does previously demoted information become causally relevant again?

## Primary hypothesis KV-H1

A frozen recoverable-tier policy will reduce task-quality loss relative to a frozen irreversible-eviction policy at matched peak live-KV bytes, without increasing the full-cache-oracle divergence beyond a preregistered tolerance.

KV-H1 is falsified if the recoverable policy fails to improve the frozen primary quality criterion, violates the matched memory budget, or its apparent gain disappears after accounting for additional compute, transfer traffic or source/context exposure.

## Secondary hypothesis KV-H2 — reactivation

Some KV segments assigned low importance at time `t` become causally relevant at a later decode step. A recoverable tier should expose this through measurable reactivation events.

KV-H2 is falsified for the tested workload if no Development-frozen reactivation metric differs from the irreversible-eviction baseline beyond the preregistered uncertainty threshold.

## Experimental arms

All arms use the same model artifact, tokenizer/template, prompts, decoding settings, maximum context and evaluation code.

- **B0 Full-cache oracle:** no KV removal or lossy tiering within the tested context.
- **B1 Irreversible eviction:** frozen selection policy removes selected segments permanently.
- **B2 Recoverable tiering:** the same selection signal moves selected segments to a lower-cost representation that remains addressable and can be promoted again.
- **B3 Optional quantized-retention ablation:** identical segment selection to B2 but a separately frozen quantization format.
- **B4 Random eviction control:** protect the same Development-frozen prompt/boundary region as the scored eviction comparison, then evict uniformly at random per attention head from the remaining eligible KV state under the same nominal live-KV budget. Freeze the RNG, seed derivation and eligibility set before Validation.

B4 is a diagnostic control, not a replacement for the primary matched B1/B2 comparison. If B1 or B2 differs from B4 in prompt protection, eligibility, per-head allocation or memory accounting, that difference must be reported rather than attributed to the selection score.

No arm may receive hidden labels or additional context unavailable to another arm.

## Intervention contract

Every KV intervention record must include:

- exact model/layer/head identity;
- token/segment/page identity;
- logical KV position range;
- intervention time before the affected decode step;
- operation: retain, demote, promote, evict, quantize, restore;
- representation format before/after;
- selection score and frozen policy version;
- logical checkpoint/lineage identity where applicable;
- physical backend identity;
- bytes resident before/after;
- bytes transferred;
- measured intervention latency when available.

A replay must fail closed when lineage or exact source identities do not match.

## Measurements

Primary measurements:

- task metric appropriate to the frozen benchmark;
- output divergence from B0 where mathematically defined;
- peak live-KV bytes;
- average live-KV bytes;
- decode latency/token and throughput;
- bytes transferred between tiers;
- number of evictions, demotions and promotions.

Reactivation measurements:

- future attention mass assigned to previously demoted/evicted segments when an oracle is available;
- a Development-frozen `Future Missed Mass` analogue measuring future full-cache attention assigned to state that an arm made unavailable;
- a Development-frozen `Global LIR` analogue measuring reactivation of previously low-importance regions, with the exact inactivity/reactivation thresholds and aggregation rule fixed before Validation;
- promotion count and promotion latency;
- quality delta attributable to restoring one selected segment or segment set using paired replay;
- false-retention and false-eviction rates under a Development-frozen relevance definition.

The QEvict-named analogues above may be used only when the full-cache attention oracle required by their frozen definitions is causally and technically available. Otherwise they are reported as unavailable rather than approximated post hoc.

No single metric may erase the quality/memory/latency trade-off.

## Paired causal replay

For a frozen checkpoint and decode state, compare paired continuations differing in exactly one declared KV intervention whenever the backend supports an exact replay contract. Preserve prompt, random seed, decoding settings and all unrelated KV state.

Metadata-only FLAT checkpoints must not be misdescribed as physical snapshots. A physical replay backend must separately prove that the required K/V bytes are preserved or reconstructed exactly enough for its declared oracle.

## Data and split discipline

- Development is used for policy choice, thresholds and metric implementation.
- Validation is evaluated after the policy is frozen.
- Any final/confirmatory split, if introduced later, remains inaccessible until separately authorized.
- A policy materially changed after Validation receives a new hypothesis/version.
- Negative and inconclusive runs are retained with provenance.

## Required baselines and fairness

B1 and B2 must use the same selection signal and nominal memory budget for the primary comparison. If B2 pays extra compute, transfer, storage or metadata cost, that cost is reported rather than hidden.

B4 must use the same protected prompt/boundary region, per-head eligibility surface and nominal live-KV budget as the scored eviction control it diagnoses. Its seed policy is frozen on Development and reused without retuning on Validation. If a scored policy fails to outperform this matched random control, that negative result is retained and the score is not credited with value unsupported by the experiment.

The full-cache oracle is a correctness/quality reference, not a deployability claim.

## Backend transfer contract

Portable semantics belong in FLAT-ATTENTION. NVIDIA-specific allocation/copy/promotion primitives belong in NNIS. KVLab may adapt either backend behind an experiment interface, but must not fork their implementation into this repository.

Any backend-specific result must be reported separately from a portable-policy result.

## Stop / falsification conditions

Stop and mark the run invalid or inconclusive if:

- exact model/workload/provenance identities are incomplete;
- the memory budgets are not matched as declared;
- intervention timing is post-hoc relative to the scored event;
- replay changes unrelated state;
- lineage validation fails;
- an arm receives additional information;
- Validation is used to retune the frozen policy;
- required quality, memory or cost metrics are missing.

## Non-claims

This preregistration does not claim that:

- retention always dominates eviction;
- attention mass is a universal measure of causal importance;
- quantized KV is equivalent to full precision;
- a FLAT logical checkpoint is a physical K/V snapshot;
- random eviction universally matches scored eviction;
- QEvict's reactivation metrics transfer unchanged to Memorithm workloads;
- any external paper result transfers to Memorithm without independent validation.
