#[path = "../src/numa_plan.rs"]
mod numa_plan;

use numa_plan::{
    MemoryPlacement, NumaCampaignError, NumaCampaignPlan, NumaCase, NumaPlanError,
    BKV_K4_NUMA_PLAN_SCHEMA_VERSION,
};

const EVIDENCE_SHA256: &str =
    "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const COMMIT_SHA: &str = "0123456789abcdef0123456789abcdef01234567";

#[test]
fn local_remote_and_interleave_cases_are_representable() {
    let local = NumaCase::new(
        "node0-local",
        vec![0, 2, 4, 6],
        0,
        MemoryPlacement::LocalNode(0),
    )
    .unwrap();
    let remote = NumaCase::new(
        "node0-remote-node1",
        vec![0, 2, 4, 6],
        0,
        MemoryPlacement::RemoteNode(1),
    )
    .unwrap();
    let interleave = NumaCase::new(
        "node0-interleave",
        vec![0, 2, 4, 6],
        0,
        MemoryPlacement::Interleave,
    )
    .unwrap();

    let plan = NumaCampaignPlan::new(EVIDENCE_SHA256, COMMIT_SHA, vec![local, remote, interleave])
        .unwrap();
    assert_eq!(plan.schema_version, BKV_K4_NUMA_PLAN_SCHEMA_VERSION);
    assert_eq!(plan.cases.len(), 3);
}

#[test]
fn duplicate_cpu_assignments_fail_closed() {
    assert_eq!(
        NumaCase::new(
            "duplicate",
            vec![0, 2, 2, 4],
            0,
            MemoryPlacement::LocalNode(0),
        ),
        Err(NumaPlanError::DuplicateCpu(2))
    );
}

#[test]
fn local_memory_must_match_declared_worker_node() {
    assert_eq!(
        NumaCase::new("bad-local", vec![1, 3], 1, MemoryPlacement::LocalNode(0),),
        Err(NumaPlanError::NodeMismatch {
            cpu_node: 1,
            placement_node: 0,
        })
    );
}

#[test]
fn remote_memory_must_differ_from_declared_worker_node() {
    assert_eq!(
        NumaCase::new(
            "bad-remote",
            vec![0, 2, 4, 6],
            0,
            MemoryPlacement::RemoteNode(0),
        ),
        Err(NumaPlanError::RemoteNodeMatchesCpuNode { cpu_node: 0 })
    );
    assert!(
        NumaCase::new(
            "node0-remote-node1",
            vec![0, 2, 4, 6],
            0,
            MemoryPlacement::RemoteNode(1),
        )
        .is_ok()
    );
}

#[test]
fn campaign_requires_evidence_commit_and_unique_case_names() {
    let case = NumaCase::new("local", vec![0], 0, MemoryPlacement::LocalNode(0)).unwrap();
    assert_eq!(
        NumaCampaignPlan::new("", COMMIT_SHA, vec![case.clone()]),
        Err(NumaCampaignError::MissingHostEvidence)
    );
    assert_eq!(
        NumaCampaignPlan::new("evidence", COMMIT_SHA, vec![case.clone()]),
        Err(NumaCampaignError::InvalidHostEvidenceSha256)
    );
    assert_eq!(
        NumaCampaignPlan::new(EVIDENCE_SHA256, "", vec![case.clone()]),
        Err(NumaCampaignError::MissingCommitSha)
    );
    assert_eq!(
        NumaCampaignPlan::new(EVIDENCE_SHA256, "commit", vec![case.clone()]),
        Err(NumaCampaignError::InvalidCommitSha)
    );
    assert_eq!(
        NumaCampaignPlan::new(EVIDENCE_SHA256, COMMIT_SHA, vec![]),
        Err(NumaCampaignError::EmptyCases)
    );
    assert_eq!(
        NumaCampaignPlan::new(EVIDENCE_SHA256, COMMIT_SHA, vec![case.clone(), case]),
        Err(NumaCampaignError::DuplicateCaseName("local".to_owned()))
    );
}

#[test]
fn provenance_identities_accept_sha1_and_sha256_git_ids() {
    let case = NumaCase::new("local", vec![0], 0, MemoryPlacement::LocalNode(0)).unwrap();
    let sha256_git_id = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    assert!(NumaCampaignPlan::new(EVIDENCE_SHA256, COMMIT_SHA, vec![case.clone()]).is_ok());
    assert!(NumaCampaignPlan::new(EVIDENCE_SHA256, sha256_git_id, vec![case]).is_ok());
}

#[test]
fn provenance_identities_reject_non_hex_and_wrong_lengths() {
    let case = NumaCase::new("local", vec![0], 0, MemoryPlacement::LocalNode(0)).unwrap();
    let bad_evidence = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg";
    assert_eq!(
        NumaCampaignPlan::new(bad_evidence, COMMIT_SHA, vec![case.clone()]),
        Err(NumaCampaignError::InvalidHostEvidenceSha256)
    );
    assert_eq!(
        NumaCampaignPlan::new("sha256:abcd", COMMIT_SHA, vec![case.clone()]),
        Err(NumaCampaignError::InvalidHostEvidenceSha256)
    );
    assert_eq!(
        NumaCampaignPlan::new(EVIDENCE_SHA256, "0123456789abcdef", vec![case.clone()]),
        Err(NumaCampaignError::InvalidCommitSha)
    );
    assert_eq!(
        NumaCampaignPlan::new(
            EVIDENCE_SHA256,
            "0123456789abcdef0123456789abcdef0123456g",
            vec![case],
        ),
        Err(NumaCampaignError::InvalidCommitSha)
    );
}

#[test]
fn sharded_and_replicated_modes_are_explicit_not_inferred() {
    let sharded = NumaCase::new(
        "sharded",
        vec![0, 2, 4, 6],
        0,
        MemoryPlacement::ShardedByNode,
    )
    .unwrap();
    let replicated = NumaCase::new(
        "replicated",
        vec![0, 2, 4, 6],
        0,
        MemoryPlacement::ReplicatedReadMostly,
    )
    .unwrap();
    assert_eq!(sharded.memory, MemoryPlacement::ShardedByNode);
    assert_eq!(replicated.memory, MemoryPlacement::ReplicatedReadMostly);
}
