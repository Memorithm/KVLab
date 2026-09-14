# ProspectEngine KV Heuristic Comparison

Schema: `kvlab.prospect-kv-heuristic-comparison/v1`

This evidence record compares one replayed synthetic KV eviction against KVLab's existing calibration heuristics under the same byte-budget ceiling:

- `oldest_first` — the exact logical eviction outcome carried by the embedded effect;
- `lru` — newest synthetic regions first;
- `magnitude` — descending contribution L2 magnitude;
- `synthetic_sensitivity_per_byte` — synthetic removal sensitivity per stored byte;
- `random` — seeded random packing with the seed recorded in the evidence.

Every policy is evaluated by the existing `evaluate_selection` additive synthetic oracle. The record stores retained region identities, retained/unused bytes and output L2 delta, then recomputes them during replay. `best_policy` means only the lowest synthetic L2 delta for that one trace and budget, with policy name as a deterministic tie-breaker.

This is a calibration comparison, not a universal policy ranking. It makes no claim about real-model attention quality, future-query utility, latency, allocator release, HBM occupancy, physical DRAM traffic, TTFT/TPOT, or production suitability. Real-model and hardware experiments remain separate gates.
