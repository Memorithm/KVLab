# Position-native comparison preflight

Real-model policy comparisons must not silently compare candidates that retain different logical KV budgets. KVLab therefore provides an explicit preflight on top of the generic position-native campaign contract.

Run:

```bash
python -m kvlab.prospect_position_comparison campaign.json
```

The input remains the existing canonical `kvlab.prospect-kv-real-model-position-campaign/v1` specification. The preflight first reuses the complete campaign validation and v2 selection handoffs, then requires every candidate policy to have the same `logical_retained_bytes` and retained-position count.

Successful output is a deterministic JSON summary containing the campaign SHA-256, ordered policy names, logical input/retained/evicted bytes, and the common retained-position count. This is a comparability gate only. The generic campaign runner remains able to execute intentionally multi-budget experiments.

The preflight does not execute a model and is not numerical, quality, latency, throughput, HBM, or memory-traffic evidence. A representative heuristic comparison still requires the campaign to be executed through an attested real-model backend and the resulting observed evidence to pass its downstream verification gates.
