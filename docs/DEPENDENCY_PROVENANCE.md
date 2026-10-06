# Dependency provenance

KVLab's executable Rust bench is qualified with Rust 1.89.0 and the checked-in
`rust/bkv_scan/Cargo.lock`. CI, fleet validation and retained campaign scripts
must use `--locked`; a dependency update is an explicit reviewed change to
that lockfile.

Every Rust CI run retains:

- the exact lockfile used by Cargo;
- `cargo metadata --locked --format-version 1`;
- SHA-256 digests for both files.

These artifacts bind the resolved graph to the source SHA and runner recorded
by GitHub Actions. They are dependency-provenance evidence only: they do not
establish model quality, GPU performance, physical memory traffic or
cross-hardware reproducibility.

Local verification:

```bash
cargo metadata --locked --manifest-path rust/bkv_scan/Cargo.toml --format-version 1
cargo clippy --locked --manifest-path rust/bkv_scan/Cargo.toml --all-targets -- -D warnings
cargo test --locked --manifest-path rust/bkv_scan/Cargo.toml
```
