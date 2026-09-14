# ProspectEngine explicit KV-selection evidence

KVLab previously exposed `kvlab.prospect-kv-eviction/v1` for one replayable logical eviction policy. That schema remains unchanged and continues to represent `oldest_first` semantics only.

Real-model heuristic comparison needs a wider structural contract because LRU, magnitude, sensitivity-per-byte, seeded-random, learned policies, and other selectors may retain different token sets under the same byte budget. Re-labeling one of those selections as `oldest_first` would destroy provenance and make replay misleading.

## `kvlab.prospect-kv-selection/v1`

The explicit-selection handoff records:

- a non-empty policy label;
- the exact ordered input token IDs;
- bytes per token;
- the exact ordered retained and evicted token IDs;
- exact logical input, retained, and evicted byte counts.

The decoder revalidates that retained and evicted IDs form a disjoint, complete partition of the input, preserve input order, and reproduce the declared logical byte accounting.

The policy label is provenance only. Structural replay proves which tokens were selected, not that an implementation of the named heuristic generated that set correctly. A claim such as “this is LRU” requires separate heuristic-specific replay or execution evidence.

## `kvlab.prospect-kv-real-model-selection/v1`

The observed evidence envelope adds:

- exact KVLab run repository revision;
- model and tokenizer revisions;
- runtime backend and runtime revision;
- evaluation/holdout identity;
- trace SHA-256 and seed;
- the embedded explicit-selection handoff;
- paired full-cache baseline and selected-candidate output-artifact SHA-256 digests;
- exact baseline/candidate logical KV byte counts;
- explicitly named numerical or quality metrics with units, preference metadata, baseline value, candidate value, and validated delta.

This schema accepts observations already produced by a backend. It does not execute a model, derive numerical or quality effects from token retention, or turn missing telemetry into zero.

## Scientific boundary

Neither schema establishes allocator release, freed HBM, avoided physical memory traffic, TTFT/TPOT reduction, throughput improvement, model-quality preservation, or production suitability. `logical_evicted_bytes` remains logical accounting. Any runtime or physical claim requires separately measured telemetry with its own provenance.

The new schemas remove the representational blocker for budget-matched real-model comparisons. They do **not** complete the representative-model milestone by themselves: that milestone remains open until actual backend executions emit comparable observed records under one controlled context and baseline.
