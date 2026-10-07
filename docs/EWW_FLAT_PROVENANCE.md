# EWW FLAT dependency provenance

Status: enforced for new evidence generated from the Rust `bkv_scan` crate.

The original EWW harnesses recorded the FLAT commit that introduced the
contract under the ambiguous `flat_revision` column.  After the direct Cargo
dependency moved forward, that historical value no longer identified the code
executed by the benchmark.

New CSV schemas separate both identities:

- `contract_origin_revision`: the immutable historical commit from which the
  benchmark contract was defined;
- `executed_dependency_revision`: the exact direct `flat-attention` revision
  compiled for the current run.

`rust/bkv_scan/build.rs` extracts the direct dependency revision from
`Cargo.toml`, requires a full 40-hex commit, and fails the build unless the
exact source is present in `Cargo.lock`.  The EWW structural smoke independently
checks every emitted CSV against that same manifest and lock identity.

Historical evidence remains bound to its original producing repository SHA and
schema.  It is not rewritten or retroactively attributed to the current FLAT
dependency.  Comparisons across v1 and v2 evidence must preserve the original
producer, contract-origin and dependency identities.

This provenance correction makes no timing, bandwidth, GPU, physical-traffic,
memory-saving or model-quality claim.
