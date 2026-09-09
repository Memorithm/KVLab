//! Core, runtime-neutral contracts for KVLab experiments.
//!
//! This bootstrap intentionally contains no CUDA, model-runtime, or framework
//! dependency. Runtime-specific capture/injection belongs behind adapters.

#![forbid(unsafe_code)]

/// Schema version for the initial KVLab bootstrap contracts.
///
/// The value remains zero until the first persisted artifact schema is defined.
pub const SCHEMA_VERSION: u32 = 0;

#[cfg(test)]
mod tests {
    use super::SCHEMA_VERSION;

    #[test]
    fn bootstrap_schema_is_unversioned() {
        assert_eq!(SCHEMA_VERSION, 0);
    }
}
