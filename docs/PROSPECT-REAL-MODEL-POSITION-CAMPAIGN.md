# Position-native real-model campaign harness

`kvlab.prospect_real_model_campaign_v4` turns the merged position-native v4
protocol into a reproducible execution boundary.

The harness accepts one **canonical JSON** campaign specification. The spec
pins the KVLab run revision, model/tokenizer/runtime identities, evaluation
identity, seed, exact model input tokens, exact teacher-forced evaluation
tokens, logical bytes per token, and explicit retained sequence positions for
each candidate policy.

Backend launch arguments are intentionally not part of the scientific spec.
They are local execution plumbing. Runtime provenance remains pinned by
`runtime_backend` and `runtime_revision`, and the v4 backend response must
attest the exact canonical request SHA-256, mode, policy, and retained
positions before KVLab accepts any observation.

## Campaign schema

Schema: `kvlab.prospect-kv-real-model-position-campaign/v1`

Example, shown pretty-printed for readability only:

```json
{
  "schema": "kvlab.prospect-kv-real-model-position-campaign/v1",
  "experiment_id": "tinyllama-position-001",
  "run_repository_revision": "<40-char KVLab revision>",
  "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "model_revision": "<pinned model revision>",
  "tokenizer_revision": "<pinned tokenizer revision>",
  "runtime_backend": "nnis-kvlab-v4",
  "runtime_revision": "<pinned NNIS revision>",
  "evaluation_id": "teacher-forced-001",
  "seed": 7,
  "bytes_per_token": 64,
  "model_input_token_ids": [7, 11, 7, 7, 19],
  "evaluation_token_ids": [23, 7],
  "selections": [
    {"policy": "lru", "retained_positions": [0, 2, 4]},
    {"policy": "magnitude", "retained_positions": [1, 2, 4]}
  ]
}
```

The file supplied to the harness must be the compact canonical form produced by
JSON key sorting with separators `(',', ':')` and no NaN/Infinity values.
Repeated vocabulary token IDs are valid. Occurrence identity is the zero-based
sequence position only.

## Execution

Use the module CLI and place the backend argv after `--`:

```text
python -m kvlab.prospect_real_model_campaign_v4 \
  --campaign campaign.json \
  --output-dir evidence/tinyllama-position-001 \
  -- \
  /path/to/nnis-kvlab-backend-v4 \
  --model /path/to/model \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --model-revision <revision> \
  --tokenizer-revision <revision> \
  --runtime-revision <NNIS revision> \
  --runtime-backend nnis-kvlab-v4
```

For the NNIS backend, provide at least two evaluation tokens: the first is the
bridge token processed after candidate KV compaction and at least one remaining
token is scored teacher-forced.

## Publication semantics

The harness performs the baseline and all candidate backend executions before
creating the evidence output directory. If a backend invocation fails,
attestation drifts, a metric is invalid, or generated evidence cannot replay,
no observed evidence output directory is published.

On success the directory contains:

- `selection-000.json`, `selection-001.json`, ...: canonical
  `kvlab.prospect-kv-real-model-selection/v2` records;
- `manifest.json`: canonical
  `kvlab.prospect-kv-real-model-position-campaign-result/v1`, containing the
  campaign-spec SHA-256, trace SHA-256, evidence schema, and SHA-256 of every
  record.

The original canonical campaign specification should be archived alongside the
result; `manifest.json` cryptographically binds it through
`campaign_spec_sha256`.

## Evidence boundary

A successful harness run means the configured backend returned observations
that passed the v4 request/response attestation and KVLab replay checks. It does
not by itself prove remote backend internals.

Logical retained/evicted bytes are not evidence of HBM release, allocator
release, avoided physical traffic, lower latency, higher throughput, or
preserved model quality. Only metrics actually measured by the executed backend
may be reported as observed results.
