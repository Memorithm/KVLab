# ProspectEngine real-model selection runner

KVLab exposes a backend-agnostic execution boundary for producing observed
`kvlab.prospect-kv-real-model-selection/v1` evidence from an actual model
runtime.

## Execution protocol

`kvlab.prospect_real_model_runner.ExternalJsonBackend` executes an argv-style
command with `shell=False`. Each invocation receives one canonical JSON request
on stdin. The backend must return one canonical JSON response on stdout.

Request schema: `kvlab.prospect-kv-backend-request/v1`.

The request binds the experiment, KVLab revision, model/tokenizer/runtime
revisions, evaluation id, trace digest, seed, exact input token ids, exact
retained token ids, bytes per token, and the policy provenance label. Baseline
requests retain the full input and carry a null policy.

Response schema: `kvlab.prospect-kv-backend-response/v1`.

The response contains:

- `output_artifact_base64`: opaque non-empty bytes representing the backend's
  evaluation artefact;
- `metrics`: explicitly named finite observations with kind, unit, preference,
  and value.

KVLab decodes the artefact and computes its SHA-256 digest itself. A backend
cannot provide the digest used by the evidence envelope. Baseline and candidate
metric names and metadata must match exactly before KVLab constructs paired
`ObservedMetric` records.

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
