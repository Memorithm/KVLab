use std::env;
use std::fs;
use std::path::PathBuf;

const DEPENDENCY_KEY: &str = "flat-attention";
const REVISION_ENV: &str = "KVLAB_FLAT_ATTENTION_EXECUTED_REVISION";

fn extract_direct_revision(manifest: &str) -> Result<String, String> {
    let prefix = format!("{DEPENDENCY_KEY} = ");
    let line = manifest
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with(&prefix))
        .ok_or_else(|| format!("missing direct dependency {DEPENDENCY_KEY}"))?;
    let marker = "rev = \"";
    let start = line
        .find(marker)
        .map(|index| index + marker.len())
        .ok_or_else(|| format!("{DEPENDENCY_KEY} must use an immutable rev"))?;
    let tail = &line[start..];
    let end = tail
        .find('"')
        .ok_or_else(|| format!("unterminated rev for {DEPENDENCY_KEY}"))?;
    let revision = &tail[..end];
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "{DEPENDENCY_KEY} rev must be a full 40-hex commit, got {revision:?}"
        ));
    }
    Ok(revision.to_owned())
}

fn main() {
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let manifest = fs::read_to_string(manifest_dir.join("Cargo.toml"))
        .expect("read bkv_scan Cargo.toml");
    let lock = fs::read_to_string(manifest_dir.join("Cargo.lock"))
        .expect("read bkv_scan Cargo.lock");
    let revision = extract_direct_revision(&manifest).expect("resolve direct FLAT revision");
    let locked_source = format!(
        "source = \"git+https://github.com/Memorithm/FLAT-ATTENTION.git?rev={revision}#{revision}\""
    );
    assert!(
        lock.lines().any(|line| line.trim() == locked_source),
        "Cargo.lock does not contain the exact direct FLAT revision {revision}"
    );

    println!("cargo:rustc-env={REVISION_ENV}={revision}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_revision_requires_a_full_commit() {
        let manifest = r#"
            [dependencies]
            flat-attention = { git = "https://example.invalid/flat", rev = "dff65361cb39b91e88a6d2bc99fc5469187809f4" }
            flat-attention-cps1 = { package = "flat-attention", rev = "ad1634fc922f6223dd3a83ac84154a82b1a35562" }
        "#;
        assert_eq!(
            extract_direct_revision(manifest).unwrap(),
            "dff65361cb39b91e88a6d2bc99fc5469187809f4"
        );
    }

    #[test]
    fn abbreviated_revision_fails_closed() {
        let manifest = r#"flat-attention = { rev = "dff65361" }"#;
        assert!(extract_direct_revision(manifest).is_err());
    }
}
