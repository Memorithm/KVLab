# SmolLM2 R2 position-native campaign preregistration

Status: **input specifications only; no observed CUDA result is recorded here**.

## Why a successor is necessary

The R1 specifications at KVLab revision
`51f2f414c6ca3ef0260d885c72b8f5863bd66047` bind NNIS revision
`58e7db8e1c4b471a7fe82a4beba11904240c4e89`. That backend preserves BF16
source tensors and then calls the F32-only decoder, which rejects the loaded
configuration. R1's pinned ProspectEngine verifier also predates a committed
Cargo.lock although its suite launcher requests a locked build. Passing R1
input-only checks did not demonstrate a working end-to-end execution.

R2 does not overwrite or relabel R1. It binds the merged NNIS #147 repair at
`091aabbb3e132627cf64716720aae530442d2a32`: source configuration remains BF16
for exact-checkpoint validation, while all actual resident execution weights
are explicitly materialized as F32 before decoder construction.

## Frozen comparison

Only `experiment_id` and `runtime_revision` differ from the respective R1
campaign inputs. Model/tokenizer snapshot, token sequence, evaluation ID, seed,
KVLab execution revision, retained positions and logical byte accounting are
unchanged. The regression tests verify that R1 bytes still match their original
SHA-256 values and prohibit other R2 input differences.

The source is `HuggingFaceTB/SmolLM2-135M` at
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, with model.safetensors SHA-256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.

The common trace is the existing 27-token prefix followed by eight evaluation
tokens: one bridge and seven scored teacher-forced targets. Each of the 7/27,
14/27 and 20/27 retained-row budgets compares `lru` and `random_seeded` with
seed 7. The declared NNIS F32 KV payload is 46,080 logical bytes per token.
The trace SHA-256 is
`3411f378fb3c7010eb94361128c019206fb47bda4271f721498d517eb07ca65f`.

These short technical campaigns cannot establish representative language-model
quality or general policy superiority. The historical R1 F16 timings are not
results of R2. No speedup, latency, throughput, HBM-release or physical-traffic
claim is authorized by this preregistration.

## Input validation

From the KVLab repository root:

```bash
python3 -m unittest discover -s tests -p 'test_smollm2_r2_position_preregistration.py'
for budget in 07 14 20; do
  python3 -m kvlab.prospect_position_comparison \
    "experiments/prospect/smollm2-r2/retain-${budget}-of-27.json"
done
```

## Execution boundary

Use the generic `kvlab.prospect_real_model_campaign_v4` executor from a clean
checkout of the declared KVLab execution revision
`404577ce939093767dc75d2d67de2fe3c16fa4dc`, with the exact R2 JSON bytes and
NNIS `091aabbb3e132627cf64716720aae530442d2a32`. Build the backend with
`cargo build --locked --release -p nnis-cli --bin nnis-kvlab-backend-v4` and
pass its exact model/tokenizer/runtime identities through the existing v4 argv
contract. Verify each resulting directory with `prospect verify-kv-campaign`
from an explicitly recorded verifier revision that contains a committed lockfile.

The fixed `prospect_smollm2_r1_suite` launcher and
`prospect verify-kv-campaign-suite` contract intentionally remain R1-only.
They must not be used by changing their constants or relabelling R2 outputs as
R1. No R2 whole-suite execution launcher or whole-suite result is supplied by
this preregistration. Individual R2 campaigns use the already existing generic
v4 execution and verification contracts.
