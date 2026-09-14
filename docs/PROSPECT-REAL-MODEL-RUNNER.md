# ProspectEngine real-model selection runner

KVLab exposes a backend-agnostic execution boundary for producing observed
`kvlab.prospect-kv-real-model-selection/v1` evidence from an actual model
runtime.

## Execution protocol

`kvlab.prospect_real_model_runner.ExternalJsonBackend` executes an argv-style
command with `shell=False`. Each invocation receives one canonical JSON request
on stdin. The backend must return one canonical JSON response on stdout.

Current request schema: `kvlab.prospect-kv-backend-request/v2`.

The request binds the experiment, KVLab revision, model/tokenizer/runtime
revisions, evaluation id, trace digest, seed, exact input token ids, exact
retained token ids, bytes per token, and the policy provenance label. Baseline
requests retain the full input and carry a null policy.

Current response schema: `kvlab.prospect-kv-backend-response/v2`.

The response contains:

- `request_sha256`: SHA-256 of the exact canonical request bytes received by the
  backend;
- `applied_mode`, `applied_policy`, and `applied_retained_token_ids`: the exact
  execution parameters the backend claims to have applied;
- `output_artifact_base64`: opaque non-empty bytes representing the backend's
  evaluation artefact;
- `metrics`: explicitly named finite observations with kind, unit, preference,
  and value.

KVLab independently computes the request SHA-256 before execution and rejects a
response whose attestation does not match the exact request, mode, policy, or
retained-token list. It also decodes the output artefact and computes the
artefact SHA-256 itself. The backend cannot choose either digest stored or
validated by the runner.

These checks make request/response binding auditable. They do **not** provide
cryptographic remote attestation of a backend's internal model execution; a
backend remains responsible for truthfully reporting which KV selection it
actually applied.

Baseline and candidate metric names and metadata must match exactly before
KVLab constructs paired `ObservedMetric` records.

## Campaign semantics

`run_real_model_selection_campaign` executes one full-cache baseline and then
one candidate per explicit selection. All selections must share the exact input
and KV geometry and must have unique policy labels.

`run_budget_matched_policy_campaign` additionally requires equal logical
retained-byte budgets across candidates. This is the execution boundary needed
for controlled LRU/magnitude/sensitivity/random/learned-policy comparisons.

## Scientific boundary

The runner is plumbing, not evidence that a representative model experiment has
already occurred. Unit tests use fixture subprocesses and therefore do not
qualify as real-model measurements.

A policy label remains provenance only. The runner does not prove that the
named heuristic generated a retained set. Logical bytes retained or evicted do
not imply allocator release, HBM reduction, physical traffic reduction,
latency improvement, TTFT/TPOT improvement, speedup, or quality preservation.
Those claims require separately measured telemetry from a real backend and an
appropriate experimental protocol.
