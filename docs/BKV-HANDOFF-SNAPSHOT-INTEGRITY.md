# Boolean-KV handoff snapshot integrity

Scope: `ProspectBkvHandoffV1`, schema `kvlab.prospect-bkv-handoff/v1`.
This is a structural correctness fix, not a new campaign, benchmark or wire format.

## Failure mode

The direct Python constructor previously accepted caller-owned lists in its
sequence fields. `@dataclass(frozen=True)` prevented attribute reassignment, but
not mutation of those lists or of nested page-word lists. Validation computed
local tuples and checked the Hamming rule, while the published object's fields
could still retain the original mutable containers. Mutating those aliases after
construction could therefore change canonical JSON without revalidating it.

The JSON decoder and cache-capture paths already supplied tuples. They retain
the same behavior and the same canonical JSON bytes for valid inputs.

## Repaired contract

Before validating content, the constructor copies query words, the page sequence,
each nested page-word sequence, and admitted IDs into tuple snapshots. Only these
snapshots are validated and retained. Tuple and list input containers are
accepted; other direct-constructor container types raise
`ProspectBkvHandoffError`. Word encodings, tail bits, geometry, integer fields,
unique/sorted IDs and exact Hamming membership retain their existing validation.
Empty admitted sets remain valid when the Hamming rule selects no page.

Caller containers are not modified. Mutating, clearing or extending them after
construction cannot change the retained signatures, candidate IDs, equality,
hash or canonical JSON. This protects ordinary API use, not hostile Python code
that deliberately bypasses a frozen dataclass with `object.__setattr__`.

## Validation

The added suite uses the real handoff constructor, packed signatures and Hamming
oracle. It covers all mutable aliases, mutable inner lists inside an immutable
outer tuple, unchanged caller containers, canonical wire compatibility,
malformed containers, incorrect candidate membership, empty selections, and
normal attempts to mutate published fields.

```bash
python3 -m unittest discover -s tests -p 'test_prospect_handoff*.py' -v
python3 -m unittest discover -s tests -v
```

The existing CI also compiles the Python package and checks the Rust Boolean-KV
scanner. All applicable jobs must pass on the final PR head before merge.
No CI result is asserted by this document; consult the corresponding run.

## Cross-project relationship

This fix follows the same validation principle as FLAT-ATTENTION MAA-9:
validate and retain the same immutable evidence, rather than accepting matching
counts or mutable metadata as proof of candidate identity. FLAT's work is tracked
in `Memorithm/FLAT-ATTENTION#235` and
`docs/research/MAA_EVIDENCE_INTEGRITY.md` in that repository.

The ownership boundary stays unchanged: KVLab owns this canonical handoff and
its Hamming oracle; FLAT owns numerical attention execution and the MAA decision
experiment. No cross-repository dependency, inference-kernel change, numerical
quality claim, timing claim or physical-memory claim is introduced here.
