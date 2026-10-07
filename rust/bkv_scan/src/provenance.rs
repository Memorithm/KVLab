//! Compile-time dependency identity for retained KVLab evidence.
//!
//! `build.rs` extracts the immutable direct `flat-attention` revision and
//! rejects a manifest/lock mismatch before any benchmark can be built.  The
//! historical revision that introduced a benchmark contract remains a
//! separate CSV field and must never be presented as the code executed now.

/// Exact direct FLAT dependency revision executed by binaries in this crate.
pub const FLAT_ATTENTION_EXECUTED_REVISION: &str =
    env!("KVLAB_FLAT_ATTENTION_EXECUTED_REVISION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executed_revision_is_a_full_immutable_commit() {
        assert_eq!(FLAT_ATTENTION_EXECUTED_REVISION.len(), 40);
        assert!(FLAT_ATTENTION_EXECUTED_REVISION
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit()));
    }
}
