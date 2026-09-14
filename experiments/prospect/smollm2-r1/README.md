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

- preregistration merge containing these campaign inputs: `51f2f414c6ca3ef0260d885c72b8f5863bd66047`;
- KVLab run revision: `404577ce939093767dc75d2d67de2fe3c16fa4dc` (budget-matched real-model position baselines available);
- NNIS runtime revision: `58e7db8e1c4b471a7fe82a4beba11904240c4e89` (exact checkpoint SHA/config admission plus fail-closed logical KV byte accounting);
- ProspectEngine verifier revision: `328dfc0c2989b9cfb2dc6b251c181141844f5241`;
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

## Pinned suite launcher

`kvlab.prospect_smollm2_r1_suite` executes the complete preregistered suite without allowing checkout drift to masquerade as the pinned provenance above. It reads the campaign JSON bytes from the preregistration merge, creates a detached KVLab worktree at the declared run revision, creates a detached NNIS worktree at the declared runtime revision, and creates a detached ProspectEngine worktree at the pinned verifier revision.

Before any CUDA execution it also:

- requires all pinned commits to exist in the supplied local repositories;
- hashes the local `model.safetensors` and requires the exact SmolLM2 digest `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`;
- builds the exact pinned NNIS v4 backend and ProspectEngine verifier;
- runs KVLab's budget-matched comparison preflight from the pinned execution worktree;
- runs ProspectEngine `verify-kv-campaign-spec` against all three exact preregistrations.

A no-GPU preflight is therefore available first:

```bash
python -m kvlab.prospect_smollm2_r1_suite \
  --nnis-repo /root/NNIS \
  --prospect-repo /root/ProspectEngine \
  --model-dir /path/to/HuggingFaceTB-SmolLM2-135M \
  --output-dir /path/to/smollm2-r1-suite \
  --device 0 \
  --preflight-only
```

Remove `--preflight-only` to execute the registered suite. Each of the three campaigns executes one full-cache baseline plus the two preregistered candidates, for nine backend invocations in total. Every campaign directory is verified with the pinned ProspectEngine binary before the suite is accepted. The complete suite is first written under a sibling staging directory and is renamed to the requested output directory only after all three campaigns and all three post-execution verifications succeed. A failure therefore does not publish a partial suite at the requested path.

The final suite directory contains the three self-contained KVLab campaign evidence directories, compact ProspectEngine verification summaries, and a canonical `suite-manifest.json` indexing exact campaign/trace hashes and pinned repository revisions.

## Manual execution sequence

The lower-level manual path remains available for a single JSON file:

1. validate it with `python -m kvlab.prospect_position_comparison <campaign.json>` from KVLab revision `404577ce939093767dc75d2d67de2fe3c16fa4dc`;
2. execute it with `python -m kvlab.prospect_real_model_campaign_v4 --campaign <campaign.json> --output-dir <new-dir> -- <nnis-kvlab-backend-v4 ...>` using NNIS revision `58e7db8e1c4b471a7fe82a4beba11904240c4e89` and the exact SmolLM2 Safetensors artifact;
3. verify the resulting self-contained directory with ProspectEngine revision `328dfc0c2989b9cfb2dc6b251c181141844f5241` via `prospect verify-kv-campaign <dir>`;
4. treat only the returned observed metrics as quality evidence for that exact execution.

No speedup, latency, throughput, HBM release, or physical traffic claim is preregistered or inferred by the suite launcher.
