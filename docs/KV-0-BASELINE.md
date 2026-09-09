# KV-0 — Baseline Reproduction and Experimental Substrate

Status: **BOOTSTRAP / NOT YET EXECUTED**

Primary source:

Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad Mahdi Kamani, Ritika Borkar, Makesh Tarun Chandran, Pantea Zardoshti, Bita Darvish Rouhani. **Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse.** arXiv:2608.03893v1, 4 Aug 2026.
https://arxiv.org/abs/2608.03893

## Purpose

KV-0 has two responsibilities:

1. establish that KVLab can capture, inject, replay, and measure KV state correctly;
2. attempt a bounded reproduction of the initial cross-model KV transfer result before extending the research space.

No reproduction result exists merely because this document exists.

## Source-supported baseline

The paper defines a source cache `C_S` and target cache `C_T` across layers and KV heads and seeks a mapping:

```text
f: C_S -> C_T_hat
```

such that the target model decoding from the mapped cache behaves approximately like the target decoding from its own cache.

For the principal production mapper, the paper uses:

- independent mappings for each target `(layer, head)`;
- separate Key and Value regressions;
- multiple selected source layers concatenated as features;
- ridge regression with `lambda = 0.01`;
- RoPE removal from source keys before mapping;
- fitting target keys in RoPE-stripped content space;
- reapplication of the target RoPE after mapping;
- calibration using 500 FineWeb-Edu sequences of length 1024, stride-4 subsampled to about 128K token observations per target head;
- a source-layer-count sweep over `1,2,4,6,8,10,12,16,20,24,all` where applicable;
- downstream functional evaluation rather than reconstruction quality alone.

The paper reports that on Qwen3 14B -> 32B, production-ridge in-sample R² rises from about `0.5572 / 0.3249` for K/V at `k=1` to `0.7914 / 0.6541` at `k=8`, with further diminishing gains when using all source layers.

It also reports that across its tested matched-KV pairs, attention-output cosine correlates with HellaSwag retention more strongly than reconstruction R², motivating subspace-aware/functional diagnostics.

## Scope boundaries inherited from the source

The source experiments used tested **matched-KV** pairs for the main transfer study, where source and target share KV-head count and per-head dimension. The paper states that mismatched-KV transfer is not tested and leaves it for future work.

The primary scope is also within-family transfer over dense full-attention models; hybrid attention and attention-recurrent architectures are outside that tested scope.

KVLab must preserve these boundaries in reproduction reports.

## KV-0A — Engineering validity gate

Before attempting cross-model mapping, validate the bench itself.

Required tests:

### A1. Exact logical geometry

Given a synthetic fixture with known:

```text
layers
kv_heads
sequence length
head_dim
dtype
```

capture and serialization must preserve every dimension exactly.

### A2. Reference replay

For a deterministic model fixture:

```text
prefill -> capture -> inject unchanged -> continue
```

must match:

```text
prefill -> continue
```

under the declared tolerance and runtime semantics.

If this fails, all transformed-cache experiments are invalid until fixed.

### A3. Identity transform

Applying `Identity` must produce the same state descriptor, payload digest where canonical storage is unchanged, and functional continuation as the unmodified reference path.

### A4. Invalid-state rejection

Test at least:

- layer out of bounds;
- head out of bounds;
- token range out of bounds;
- feature range out of bounds;
- incompatible target geometry;
- wrong dtype where the adapter requires exact dtype;
- incompatible positional metadata when required.

### A5. Manifest completeness

A run must fail validation if required provenance fields are absent.

## KV-0B — Linear-structure probe

Before the production mapper, implement a single-source-layer probe mirroring the source paper's analysis.

For each compatible source layer, target layer, and target head:

- fit an ordinary least-squares or equivalent declared linear probe;
- evaluate Key and Value separately;
- evaluate raw/RoPE-coupled keys and RoPE-stripped keys where supported;
- retain the full layer-to-layer score matrix;
- report head-averaged R² without discarding per-head values.

Purpose:

- verify that the chosen model pair exhibits or does not exhibit the expected linear structure;
- inspect diagonal/banded structure;
- determine whether RoPE factoring materially changes the fit in the reproduced environment;
- avoid building a complex mapper on a pair where the baseline relationship does not reproduce.

## KV-0C — Multi-source ridge mapper

Implement the source-paper baseline as faithfully as practical.

For each target layer:

1. score candidate source layers using the declared selection protocol;
2. choose the top `k` source layers;
3. concatenate their K features and V features separately;
4. center `X` and `Y`;
5. solve ridge regression;
6. recover the bias from the means;
7. store mapping parameters with exact source/target geometry metadata;
8. apply K mapping in content space and reapply target RoPE;
9. apply V mapping directly;
10. inject the mapped target cache and evaluate continuation.

Do not mix K/V parameters or silently share parameters across target heads unless an explicit extension experiment is being tested.

## KV-0D — Functional evaluation

The first reproduction should prioritize a small, feasible, well-instrumented subset rather than pretending to reproduce the paper's entire compute budget.

At minimum record:

- target standalone behavior;
- mapped-cache behavior;
- source standalone behavior where relevant;
- tensor reconstruction metrics;
- next-token/logit diagnostics;
- at least one downstream functional metric suitable for the chosen models;
- transform latency;
- target re-prefill latency under the same measurement discipline.

When expanding toward the paper's evaluation suite, relevant reported benchmarks include ARC-Challenge, HellaSwag, WinoGrande, MMLU, GSM8K, WikiText-2 prefix-conditioned perplexity, and CoQA for multi-turn handoff.

## KV-0E — Mechanism analysis

When backend observability permits it, reproduce the paper's distinction between residual magnitude and residual placement.

Candidate analysis:

1. capture the target queries for a declared layer/head/sample set;
2. compute a declared basis of query-visible directions;
3. project Key mapping residuals onto strongly and weakly observed directions;
4. compare residual concentration with attention-output change and downstream retention;
5. compare these relationships with global R²/MSE.

Do not assume the paper's reported correlation value will reproduce on a smaller model/sample domain.

## Initial model strategy

Prefer an execution sequence that minimizes infrastructure risk:

1. tiny deterministic/random fixture for capture/injection correctness;
2. small openly available same-family model pair for mapper plumbing;
3. one source-paper family/pair that is feasible on available hardware;
4. only then broaden to additional paper pairs or later KV series.

The exact model pair must be selected from actual available hardware/model access and recorded in the experiment manifest. Do not invent a reproduction claim for an unexecuted pair.

## Result classification

At the end of the first real KV-0 reproduction attempt, classify the outcome as one of:

- `REPLICATED_WITHIN_DECLARED_TOLERANCE`;
- `PARTIALLY_REPLICATED`;
- `NOT_REPLICATED_UNDER_TESTED_CONDITIONS`;
- `INCONCLUSIVE`;
- `INVALID_RUN`.

The report must state exactly which source-paper observations were included in the decision.

## KV-0 exit condition

KV-0 is complete only when:

- the capture/injection reference gate passes;
- the transform/provenance substrate is deterministic and tested;
- at least one real model-pair mapping experiment has been executed;
- the result and raw metrics are committed or referenced reproducibly;
- discrepancies from the source paper are documented;
- later-series experiments can reuse the substrate without one-off cache surgery code.
