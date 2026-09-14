# Position-native real-model baselines

KVLab separates real-model position controls from its synthetic contribution-vector baselines.

`kvlab.prospect_position_baselines` provides two deterministic controls that need no fabricated model signature:

- `lru`: retains the newest sequence positions under an exact row budget;
- `random_seeded`: retains a reproducible seeded random subset under the same exact row budget and returns positions in canonical sorted order.

`budget_matched_lru_random(...)` constructs both controls at one retained-position count. The comparison preflight can then verify that their logical byte budgets are identical once they are embedded in a canonical position campaign.

Magnitude and sensitivity selectors from the synthetic calibration modules are intentionally not exposed here. A real-model magnitude/sensitivity policy requires an observed or otherwise explicitly defined real-model signature and must not be inferred from synthetic contribution vectors.

These selectors produce candidate positions only. They are not execution evidence and do not establish quality preservation, allocator release, HBM savings, avoided traffic, latency, throughput, or general superiority of one policy.
