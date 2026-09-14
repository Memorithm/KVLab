# ProspectEngine real-model KV eviction evidence

Schema: `kvlab.prospect-kv-real-model-eviction/v1`.

This contract is the observed-evidence boundary between a backend that has actually executed a model and ProspectEngine. KVLab validates and serializes those observations; it does not execute the model in this module and it never derives quality, numerical drift, latency, physical traffic, or memory release from logical eviction bytes.

A record binds:

- the exact KVLab run repository revision;
- model, model revision, tokenizer revision, runtime backend and runtime revision;
- evaluation/holdout identity, trace SHA-256 and seed;
- the canonical replayable `kvlab.prospect-kv-eviction/v1` logical eviction;
- baseline and candidate output-artifact SHA-256 digests;
- baseline and candidate logical KV bytes, which must exactly match the embedded eviction input/retained accounting;
- one or more observed numerical or quality metrics with explicit units, preference metadata and replayable `candidate - baseline` deltas.

The schema deliberately does not define a universal quality metric or promotion rule. A consumer may apply a preregistered policy to observed metrics, but the evidence record itself only states what was measured.

## Execution gate

The existence of this schema is **not** evidence that a real-model run has occurred. Milestone completion requires canonical records emitted from an actual backend execution under a frozen model/runtime/trace configuration. Fixture tests validate the evidence machinery only.

## Claims that remain out of scope

Without separately measured fields, no claim is made about allocator release, freed HBM, physical DRAM traffic, host-device transfers, latency, TTFT/TPOT, throughput, or production suitability. Logical KV bytes remain logical accounting only.
