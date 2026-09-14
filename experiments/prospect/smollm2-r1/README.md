# SmolLM2 R1 position-native campaign preregistration

This directory contains input specifications only. No file here is observed campaign evidence.

## Fixed trace source

The token trace is copied exactly from `Memorithm/NNIS/evidence/r1_nnis_f16_smollm2_thor.json`:

- source model: `HuggingFaceTB/SmolLM2-135M`;
- immutable source revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- prompt: `Gravity is`;
- prompt IDs: `[22007, 6463, 314]`;
- the following 32 token IDs are the exact R1 greedy trajectory recorded by NNIS.

For these campaigns, the prompt plus the first 24 generated IDs form a 27-position model prefix. The remaining 8 generated IDs are the evaluation trace; the first evaluation ID is the v4 bridge token and the final 7 IDs are teacher-forced scored targets.

The R1 performance measurements and its f16 execution plan are **not** reused as evidence for these campaigns. Only the explicit token trace is reused. Candidate and baseline quality must be measured again through the pinned v4 backend.

## Pinned execution contracts

- KVLab run revision: `404577ce939093767dc75d2d67de2fe3c16fa4dc` (budget-matched real-model position baselines available);
- NNIS runtime revision: `58e7db8e1c4b471a7fe82a4beba11904240c4e89` (exact checkpoint SHA/config admission plus fail-closed logical KV byte accounting);
- runtime backend: `nnis-kvlab-v4`;
- exact NNIS logical KV payload for this model: `46,080` bytes/token (`KvCache<f32>`);
- campaign seed: `7`.

The tokenizer revision is pinned to the same immutable upstream snapshot as the model. NNIS's SmolLM2 fixture resolves `tokenizer.json` from that exact snapshot and computes its SHA-256 from bytes on disk. The historical R1 evidence file itself does not include that tokenizer digest, so these preregistrations do not claim that R1 carries a tokenizer-artifact hash attestation. The v4 backend consumes token IDs and does not load a tokenizer.

## Budget-matched controls

Three independent campaign inputs are preregistered:

- retain 7 of 27 positions;
- retain 14 of 27 positions;
- retain 20 of 27 positions.

Each campaign compares exactly two position-only controls at the same row/byte budget:

- `lru`: newest positions;
- `random_seeded`: deterministic random subset under seed 7.

Magnitude and sensitivity policies are deliberately absent. KVLab's current magnitude/sensitivity baselines are synthetic and cannot be relabeled as real-model signatures.

## Required execution sequence

For each JSON file:

1. validate it with `python -m kvlab.prospect_position_comparison <campaign.json>`;
2. execute it with `python -m kvlab.prospect_real_model_campaign_v4 --campaign <campaign.json> --output-dir <new-dir> -- <nnis-kvlab-backend-v4 ...>` using the exact pinned NNIS revision and exact SmolLM2 Safetensors artifact;
3. verify the resulting self-contained directory with ProspectEngine `prospect verify-kv-campaign <dir>`;
4. treat only the returned observed metrics as quality evidence for that exact execution.

No speedup, latency, throughput, HBM release, or physical traffic claim is preregistered here.
