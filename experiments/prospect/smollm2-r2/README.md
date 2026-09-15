# SmolLM2 R2 position-native campaign preregistration

Status: **input specifications and execution tooling; no observed CUDA result is recorded here**.

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
python3 -m unittest discover -s tests -p 'test_smollm2_r2*.py'
python3 -m unittest discover -s tests -p 'test_prospect_r2_publication_gate.py'
for budget in 07 14 20; do
  python3 -m kvlab.prospect_position_comparison \
    "experiments/prospect/smollm2-r2/retain-${budget}-of-27.json"
done
```

## R2 suite launcher

`python3 -m kvlab.prospect_smollm2_r2_suite` uses five distinct immutable pins:

| Role | Revision |
| --- | --- |
| Preregistered R2 JSON bytes | KVLab `216b49ae4d62ed4c4c2edfd1e88f929d0a0fd9e5` |
| Actual v4 campaign executor | KVLab `404577ce939093767dc75d2d67de2fe3c16fa4dc` |
| F32 NNIS backend | NNIS `091aabbb3e132627cf64716720aae530442d2a32` |
| Historical per-campaign verifier | ProspectEngine `298acdc91682ef1d09914b6f964e8934828825c0` |
| Additional global publication verifier | ProspectEngine `ca9685cd98f3a0a23e8c4f7e368736bb3aa28d0c` |

The launcher reuses the existing streamed model hashing and subprocess utilities;
it does not mutate R1 constants, copy model execution code, modify the caller's
checkout, or change any preregistered JSON. Each input's exact SHA-256 is checked
before parsing. Repository commits must already be present in the local clones;
the launcher does not fetch or install dependencies on the user's behalf. Cargo
may download the dependencies declared by the committed lockfiles.

Install the explicit toolchain first:

```bash
rustup toolchain install 1.89.0 --profile minimal
```

With `NNIS_REPO`, `PROSPECT_REPO`, `MODEL_DIR`, and `OUTPUT_DIR` set to the actual
local paths, run the no-CUDA readiness check from the KVLab repository root:

```bash
python3 -m kvlab.prospect_smollm2_r2_suite \
  --nnis-repo "${NNIS_REPO:?Set NNIS_REPO}" \
  --prospect-repo "${PROSPECT_REPO:?Set PROSPECT_REPO}" \
  --model-dir "${MODEL_DIR:?Set MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR:?Set a new OUTPUT_DIR}" \
  --preflight-only
```

Preflight verifies commits and the local model digest, builds all three binaries
with Rust 1.89 and `--locked --release`, and runs the two pinned input verifiers
against all three campaign files. It does not invoke the NNIS backend, initialize
CUDA, create an observed result, run global observed-suite verification, or create
the requested output directory. The compatible preflight schema remains
`kvlab.smollm2-r2-position-suite-preflight/v1`.

Removing `--preflight-only` explicitly requests actual CUDA model execution. The
existing pinned executor runs one baseline and two candidates per budget. Each
complete campaign directory is independently checked by the historical pinned
ProspectEngine binary; the launcher also checks exact input identities, positions
and budgets in the returned summary. After writing the full suite manifest it
must pass the additional `verify-kv-campaign-suite-r2` check, including common
baseline agreement across budgets, before the staged suite can be renamed to the
new output directory. There is no global-verification bypass flag.

The result schema remains `kvlab.smollm2-r2-position-suite-result/v1`. It preserves
the per-campaign directories, verification summaries and `suite-manifest.json`.
The historical verifier field in that manifest is not replaced. Diagnostics and
the additional global-check receipt are written to stderr; stdout is reserved for
the original final JSON response. Preserve the log outside the strict suite file
set. A `stage_verified` receipt is not an assertion that final publication or a
GPU execution has been independently authenticated.

Errors, global-verification rejection and interrupts remove the staged result.
Existing output entries, including dangling symlinks, are rejected. A cooperative
sibling-directory lock prevents two launches from using the same destination.
Python execution uses the pinned worktree and ignores inherited Python path/home
and user-site settings. The binary and directories must be trusted and unmodified;
this is neither an adversarial filesystem sandbox nor a power-loss durability claim.

The fixed `prospect_smollm2_r1_suite` launcher and
`prospect verify-kv-campaign-suite` remain R1-only. R2 results must not be relabelled
as R1. See [the publication gate contract](../../../docs/PROSPECT-R2-PUBLICATION-GATE.md)
for separate verifier provenance, failure handling and receipt limits. Mocked or
synthetic integration tests are not model executions or measured quality evidence.
