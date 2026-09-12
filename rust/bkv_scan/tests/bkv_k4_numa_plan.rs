#[path = "../src/numa_plan.rs"]
mod numa_plan;

use numa_plan::{
    BKV_K4_NUMA_PLAN_SCHEMA_VERSION, MemoryPlacement, NumaCampaignError, NumaCampaignPlan,
    NumaCase, NumaPlanError,
};

#[test]
fn local_remote_and_interleave_cases_are_representable() {
    let local = NumaCase::new("node0-local", vec![0, 2, 4, 6], 0, MemoryPlacement::LocalNode(0))
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

    let plan = NumaCampaignPlan::new(
        "sha256:host-evidence",
        "commit-sha",
        vec![local, remote, interleave],
    )
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
        NumaCase::new(
            "bad-local",
            vec![1, 3],
            1,
            MemoryPlacement::LocalNode(0),
        ),
        Err(NumaPlanError::NodeMismatch {
            cpu_node: 1,
            placement_node: 0,
        })
    );
}

#[test]
fn campaign_requires_evidence_commit_and_unique_case_names() {
    let case = NumaCase::new("local", vec![0], 0, MemoryPlacement::LocalNode(0)).unwrap();
    assert_eq!(
        NumaCampaignPlan::new("", "commit", vec![case.clone()]),
        Err(NumaCampaignError::MissingHostEvidence)
    );
    assert_eq!(
        NumaCampaignPlan::new("evidence", "", vec![case.clone()]),
        Err(NumaCampaignError::MissingCommitSha)
    );
    assert_eq!(
        NumaCampaignPlan::new("evidence", "commit", vec![]),
        Err(NumaCampaignError::EmptyCases)
    );
    assert_eq!(
        NumaCampaignPlan::new("evidence", "commit", vec![case.clone(), case]),
        Err(NumaCampaignError::DuplicateCaseName("local".to_owned()))
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
