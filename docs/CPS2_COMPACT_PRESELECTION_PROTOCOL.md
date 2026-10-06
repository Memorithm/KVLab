# CPS-2 compact-preselection comparative quality protocol

Status: preregistered **development/synthetic** protocol, 2026-09-30. This is not
a protected holdout, real-model result, physical-performance result, or runtime
promotion gate.

## Source and producer identity

The motivating discussion proposed a low-dimensional K/Q screen followed by
exact attention on retained keys, plus sparse traversal and online softmax.
KVLab treats that discussion as a hypothesis source only; it does not assume the
described "CSA-2" properties, SIMD/FMA timing, cache behavior, or memory claims
are established facts.

The executable producer is the independently qualified FLAT CPS-1 implementation
merged as:

```text
repository: Memorithm/FLAT-ATTENTION
merge revision: ad1634fc922f6223dd3a83ac84154a82b1a35562
producer API: flat_algebraic_attention::compact_preselection::compact_preselect
consumer API: flat_attention::api::research_structural_routing::forward_reference_structural_sparse
```

KVLab must pin that exact revision for this protocol. A later producer revision
requires a new evidence identity and rerun.

## Question

For a fixed candidate budget, does coordinate-projected preselection preserve
reference attention structure better than simple matched-density controls while
reducing **logical selector score components** relative to a full-dimensional
ranking control?

This protocol does not ask whether the mechanism is faster. It does not equate
selected numerical pairs with time or physical traffic.

## Frozen arms

Each row is evaluated over the causally eligible original key positions.

1. `all_accept` — all eligible keys; numerical correctness control.
2. `compact_projected` — FLAT CPS-1 coordinate-projected top-k producer.
3. `full_score_topk` — the same FLAT producer with every head coordinate
   retained; evaluation upper bound for ranking at the same candidate budget.
4. `recent_tail` — latest eligible keys at exactly the compact selected count.
5. `matched_random` — deterministic SplitMix64 row/page ranking at exactly the
   compact selected count.

The random ranking identifier is:

```text
splitmix64-row-page-ranking-v1
```

For row `r`, derive `row_seed = splitmix64(seed XOR r)` and rank key `j` by
`splitmix64(row_seed XOR j)`, then by the original key id. Selected ids are
canonicalized into increasing original-position order before numerical
attention.

## Frozen development metrics

Record separately for every row and arm:

- selected original key ids;
- causally eligible key count;
- selected density;
- reference top-k ids;
- top-k hits and recall;
- retained and omitted dense-reference softmax mass;
- maximum absolute output error versus dense FLAT;
- absolute LSE error versus dense FLAT;
- selector score-component count;
- exact numerical query-key pairs executed by the structural sparse oracle.

Do not collapse these metrics into one score.

`retained_softmax_mass` is an evaluation oracle using exact full-dimensional
reference scores. It is not a deployable confidence signal unless a later
protocol proves a deployable equivalent.

## Logical accounting boundary

The compact producer reports:

```text
projected_key_payload_bytes
evaluated_pairs
evaluated_score_components
selected_pairs
```

These are logical representation/work counts. They exclude, as applicable,
retained full numerical K/V, allocator overhead, cache-line transactions,
physical DRAM/HBM traffic, transfers, synchronization, launch cost, and the
downstream numerical attention work. They must not be converted into speedup or
bandwidth claims.

The full numerical K/V remains retained and authoritative in CPS-2.

## Frozen synthetic fixtures

The CI development panel uses two N=5, D=2 fixtures, candidate budget 2,
reference top-k 2, seed `0x43505332`, and non-causal attention.

### aligned_coordinate

```text
Q row = [1, 0]
K_j   = [j, 0]
V_j   = [j, 4-j]
projection coordinates = [0]
```

This checks the case where the compact coordinate contains the ranking signal.

### omitted_dominant_coordinate

```text
Q row = [1, 1]
K_0   = [0, 20]
K_j   = [j, 0] for j > 0
V_j   = [j, 4-j]
projection coordinates = [0]
```

This is a mandatory falsification control. The compact arm is expected to miss
the dominant key. Exact numerical scoring of the retained set cannot repair a
false negative that the selector has already removed.

No parameter is to be retuned to make this negative fixture pass.

## Robustness tests

The Rust suite must cover:

- exact all-accept O/LSE parity with dense FLAT;
- irregular N=19 / D=17 geometry;
- causal filtering before selection;
- exact matched density for recent and random controls;
- deterministic seed-bound random selection;
- full-dimensional top-k control recall;
- separate compact versus full selector accounting;
- invalid zero budget / zero reference-k / malformed projection rejection;
- exact source revision and evidence schema binding.

## Execution

The repository CI gate for this slice runs:

```bash
cargo fmt --manifest-path rust/bkv_scan/Cargo.toml -- --check
cargo clippy --locked --manifest-path rust/bkv_scan/Cargo.toml --all-targets -- -D warnings
cargo test --locked --manifest-path rust/bkv_scan/Cargo.toml
```

The Rust tests include deterministic repeated-panel equality under a frozen
seed. The CSV runner is an explicit evidence command, not an implicit CI claim:

```bash
cargo run --locked --manifest-path rust/bkv_scan/Cargo.toml --bin cps2_compact_panel
```

If CSV output is retained as evidence, run the command twice on the same exact
revision and require byte-identical output before archiving it.

## Promotion boundary

CPS-2 synthetic evidence cannot change FLAT, SLHAv2, ElasticXxx, SciRust or NNIS
defaults.

The next scientific gate requires frozen real-model/model-tokenizer-dataset
identities and data separation before execution. Candidate coordinates,
projection policy, budget, reference-k, seeds and acceptance criteria must be
frozen before any protected population is observed.
