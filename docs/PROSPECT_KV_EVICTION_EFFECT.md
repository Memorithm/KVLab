# ProspectEngine KV Eviction Effect Contract

Schema: `kvlab.prospect-kv-eviction-effect/v1`

This contract connects two existing KVLab components without collapsing their meanings:

1. the exact logical `oldest_first` eviction semantics from `kvlab.eviction` / `kvlab.prospect_eviction_handoff`;
2. the additive synthetic numerical oracle from `kvlab.synthetic_trace` / `kvlab.evaluation`.

The envelope embeds the ordered token-to-region binding, the complete synthetic trace contribution vectors, the replayable logical eviction handoff, the retained and evicted region identities, the full-cache output, the retained-cache output, and the resulting L2 output delta.

## Validation rules

A valid record must satisfy all of the following:

- every logical input token maps to exactly one synthetic region;
- every synthetic region is covered exactly once;
- bound-region order matches the eviction input-token order;
- every bound region has `storage_bytes == eviction.bytes_per_token`;
- the embedded logical eviction independently replays through KVLab;
- the retained-region set is derived from the replayed retained token IDs;
- the numerical output is recomputed by the existing synthetic evaluation oracle;
- canonical JSON round-trips without unknown fields or non-finite numbers;
- recorded outputs, L2 delta, and logical evicted bytes must match replay.

## Scientific boundary

This is an executed **synthetic** numerical experiment, not real-model evidence. It can establish that a precisely specified logical eviction changes the additive synthetic oracle by a precisely reproducible amount. It does not establish model-quality preservation, allocator release, freed GPU/HBM memory, latency reduction, transfer avoidance, physical DRAM-traffic reduction, TTFT/TPOT improvement, or production suitability.

Those claims require independent real-model/hardware evidence and must not be inferred from `logical_evicted_bytes` or the synthetic L2 delta.
