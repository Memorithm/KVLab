# K9 protocol integrity

This implementation note strengthens the existing cross-model KV transfer prerequisites. It does not change a preregistered experiment, fit a mapper, evaluate a model or establish transfer quality.

## Immutable capture

`CrossModelTransferProtocol` now accepts explicit list/tuple split IDs, validates immutable tuple snapshots, and retains those exact snapshots. Altering an input list after construction cannot introduce calibration/final-holdout overlap into the validated protocol. Ordering and string contents are preserved without sorting, trimming or renaming identities. Empty, duplicate or overlapping identities remain errors. Strings, mappings, unordered sets and one-shot iterators are rejected as split containers rather than ambiguously interpreted.

## Runtime type and numeric validation

- Geometry and source-layer counts must be positive Python integers, not booleans or floats.
- Model metadata requires non-empty strings and validated geometry objects; protocol roles require model revision objects.
- `mapper` requires a `MapperBaseline` enum member. Raw `"ridge"` must not bypass an identity-based ridge/RoPE check.
- `remove_rope_from_keys` requires an actual boolean, not a truthy string or integer.
- `ridge_alpha` requires a finite non-negative Python int/float, excluding booleans and integer magnitudes that overflow the finite-float check. Existing zero-alpha allowance and explicit linear ablation behavior are preserved. Valid numeric values are not coerced into a different type.

Malformed input raises `ValueError` during construction. This is defensive validation for ordinary callers, not a sandbox against hostile Python code that deliberately circumvents frozen dataclasses.

## Regression coverage

The six existing protocol tests are retained, with twelve added integrity tests. The tests exercise caller-list mutation, exact identity order/content, invalid containers and entries, non-finite/invalid numeric inputs, geometry/layer types, mapper/RoPE bypass attempts, malformed model metadata and preserved valid baselines. The complete repository CI also checks downstream K9 modules and the Rust Boolean scanner.

## Shared engineering lesson with FLAT

FLAT-ATTENTION PR #234 retains immutable view/tier metadata before comparing page-tier assignments. K9 uses the same capture-then-validate principle for experimental split identities, without copying FLAT's physical page logic or conflating its process-local cache provenance with cross-model transfer evidence.

The K9 applicability decision still expresses the existing protocol prerequisites only. It is not authorization to open a final holdout, a fitted transfer model, proof of tokenizer alignment, or a claim that arbitrary architectures can exchange KV state. Those require their own frozen capture, alignment, evaluation and provenance gates.
