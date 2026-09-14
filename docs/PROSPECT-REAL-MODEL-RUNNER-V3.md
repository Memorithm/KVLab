# ProspectEngine real-model runner protocol v3

Protocol v3 removes an ambiguity that prevents a rigorous model backend from using the original selection handoff directly.

`ProspectKvSelectionHandoffV1.input_token_ids` are **logical KV row identities**. They are unique by contract so an exact retained/evicted partition can be replayed. They must not be assumed to be model vocabulary token ids: real tokenized sequences may contain the same vocabulary id multiple times.

## Canonical trace

`kvlab.prospect-kv-real-model-trace/v1` binds three ordered arrays:

- `logical_input_token_ids`: unique identities for the prefill KV rows;
- `model_input_token_ids`: actual vocabulary ids, position-aligned one-to-one with the logical identities and allowed to repeat;
- `evaluation_token_ids`: a non-empty actual-token continuation used after the baseline or candidate history has been prepared. These ids may also repeat.

The trace is serialized as canonical JSON and SHA-256 hashed. `RealModelRunContext.trace_sha256` must equal that digest before any backend process is executed.

## Backend request

Request schema: `kvlab.prospect-kv-backend-request/v3`.

The request carries the complete canonical trace data, the explicit `retained_logical_token_ids`, the policy provenance label, byte accounting and the existing model/runtime/evaluation provenance. Baseline retains every logical input row. Candidate requests contain the exact logical retained subset.

Response schema: `kvlab.prospect-kv-backend-response/v3`.

The backend must attest:

- SHA-256 of the exact canonical request;
- applied baseline/candidate mode;
- applied policy label;
- exact applied retained logical identities;
- one opaque non-empty output artefact;
- a non-empty set of finite named observed metrics.

KVLab computes the artefact SHA-256 itself and pairs baseline/candidate metrics only when name, kind, unit and preference agree exactly.

## Intended execution semantics

A concrete autoregressive backend can use the trace as follows:

1. prefill `model_input_token_ids` in their original order;
2. map each `retained_logical_token_id` to its position in `logical_input_token_ids`;
3. for a candidate, physically select/compact those KV rows without renumbering their already encoded positional representation;
4. teacher-force the identical `evaluation_token_ids` continuation for baseline and candidate;
5. emit directly observed metrics such as mean target negative log-likelihood and top-1 token accuracy, plus an artefact sufficient for replay/audit.

The protocol does not require these particular metric names, but a campaign must compare like-for-like metadata.

## Scientific boundary

Logical retained bytes remain logical accounting. They do not establish allocator release, smaller fixed cache capacity, reduced HBM reservation, reduced traffic, lower latency, higher throughput or preserved model quality.

Fixture subprocesses in unit tests validate only the protocol. They are not representative model experiments. A representative quality or performance conclusion requires a real model/runtime, a declared evaluation trace and separately appropriate measurement methodology.
