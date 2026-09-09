# KVLab Scientific Contract

This document defines how KVLab converts code execution into scientific evidence.

## 1. Object of study

KVLab studies an autoregressive model's KV state as a structured intermediate representation.

A cache artifact must be described at least by:

```text
model identity
model revision
runtime/backend
tokenizer identity
sequence length
layer count
KV head count
head dimension
dtype / numerical policy
positional-encoding metadata where relevant
K and V storage geometry
capture point
```

Do not assume all runtimes store KV in the same physical layout. KVLab's logical schema must distinguish logical geometry from backend-specific storage layout.

## 2. Experimental unit

A minimal experiment consists of:

```text
Reference input
+ Reference model/runtime state
+ Captured KV artifact
+ Explicit transform plan
+ Injected/continued execution
+ Reference execution
+ Declared metrics
+ Provenance
= Experiment record
```

An intervention without a reference execution is an engineering test, not sufficient scientific evidence.

## 3. Claim classes

Every reported conclusion must use one of the following classes:

- `OBSERVATION` — directly measured result under recorded conditions;
- `HYPOTHESIS` — proposed explanation or prediction not yet established;
- `SUPPORTED` — preregistered or otherwise declared claim whose decision rule passed under the tested conditions;
- `REFUTED` — declared claim whose decision rule failed under the tested conditions;
- `INCONCLUSIVE` — evidence cannot distinguish the tested alternatives or execution validity is insufficient;
- `ENGINEERING_ONLY` — implementation property such as serialization round-trip, shape validation, or runtime compatibility.

No class implies universality beyond the recorded experiment domain.

## 4. Reproduction before extension

KV-0 begins from Heo et al., arXiv:2608.03893v1.

The paper reports, among other findings, that:

- cross-model KV transfer can exhibit substantial linear structure for tested matched-KV pairs;
- multiple source layers provide complementary predictive information;
- the production method uses independent per-target-head ridge mappings for K and V;
- keys are mapped in RoPE-stripped content space and target RoPE is reapplied;
- downstream functional retention is not reliably predicted by reconstruction R² alone;
- attention-output similarity is more informative across the tested transfer pairs;
- the paper does not test mismatched-KV pairs and restricts its main scope to within-family dense full-attention models.

Source: https://arxiv.org/abs/2608.03893

KVLab must first reproduce or explicitly fail to reproduce a bounded subset of these claims before treating them as local baseline assumptions.

## 5. Metrics are not interchangeable

Separate the following categories:

### 5.1 Representation metrics

Examples:

- MSE;
- relative L2;
- cosine similarity;
- R²;
- singular-spectrum change.

These describe tensor relationships.

### 5.2 Mechanism metrics

Examples:

- attention-output cosine;
- attention score/probability divergence;
- query-visible error energy;
- per-head activation difference.

These describe how the transformed state interacts with model computation.

### 5.3 Functional metrics

Examples:

- next-token agreement;
- logit divergence;
- perplexity under a declared protocol;
- benchmark/task accuracy;
- multi-turn behavior.

These describe model behavior.

### 5.4 Systems metrics

Examples:

- transform latency;
- prefill latency avoided;
- decode latency;
- GPU memory;
- host memory;
- transfer volume;
- synchronization cost.

A result must not substitute one category for another without an explicit argument and validation.

## 6. Controls

Every intervention study must include the controls needed to isolate its effect.

Typical controls include:

- unmodified reference cache;
- capture -> inject round-trip with no transform;
- identity transform;
- intervention-size-matched random control;
- parameter-count/compute-matched mapper baseline;
- seed repetitions where stochasticity exists;
- source/target standalone executions for transfer experiments.

## 7. Selection and holdouts

Any hyperparameter search, mapper selection, transform search, or policy optimization must identify:

- calibration/training data;
- development/validation data;
- final holdout data, if used;
- selection criterion;
- tie-breaking rule;
- stopping rule.

Final holdout evidence must not be exposed to an automated search loop.

If a decision rule changes after observing the target evidence, the result becomes exploratory and must be labeled accordingly.

## 8. Reproducibility levels

KVLab should progressively classify runs using explicit reproducibility levels, for example:

- `R0` — prose description only; insufficient for a scientific result;
- `R1` — code revision and command recorded;
- `R2` — inputs/models/runtime versions and deterministic parameters recorded;
- `R3` — machine-readable manifest plus artifact hashes and raw metrics;
- `R4` — independently reproduced on the declared environment;
- `R5` — reproduced across an additional declared environment/model/sample domain.

Do not confuse reproducibility level with truth of a hypothesis.

## 9. Failure handling

A run is invalid rather than negative when, for example:

- the reference cache itself does not replay correctly;
- model/tokenizer identity is ambiguous;
- source and target sequences are not aligned as required by the method;
- an intervention writes out of declared geometry;
- required hardware execution is silently skipped;
- a metric implementation fails its oracle tests;
- the experiment mixes incompatible positional conventions without accounting for them.

Invalid runs must be retained as engineering/debug artifacts when useful but excluded from scientific decision rules.

## 10. Novelty policy

KVLab may discover new empirical phenomena or algorithms, but novelty must not be inferred from absence of local knowledge.

Before a repository result is described as novel:

1. reproduce it;
2. falsify obvious alternative explanations;
3. search the current literature and implementations;
4. distinguish the exact new contribution from known methods;
5. state the tested domain and failure cases.

Until then use neutral language such as `candidate method`, `observed effect`, or `working hypothesis`.
