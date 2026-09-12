#[path = "../src/numa_evidence.rs"]
mod numa_evidence;

use numa_evidence::{NumaEvidenceError, NumaHostEvidence};

#[test]
fn worker_sets_are_validated_against_declared_topology() {
    let evidence = NumaHostEvidence::new([(0, 0), (2, 0), (1, 1), (3, 1)]).unwrap();

    assert_eq!(evidence.node_for_cpu(2), Some(0));
    assert!(evidence.contains_node(1));
    assert_eq!(evidence.validate_worker_set(&[0, 2], 0), Ok(()));
    assert_eq!(evidence.validate_worker_set(&[1, 3], 1), Ok(()));
}

#[test]
fn unknown_and_wrong_node_cpus_fail_closed() {
    let evidence = NumaHostEvidence::new([(0, 0), (2, 0), (1, 1)]).unwrap();

    assert_eq!(
        evidence.validate_worker_set(&[0, 4], 0),
        Err(NumaEvidenceError::UnknownCpu(4))
    );
    assert_eq!(
        evidence.validate_worker_set(&[0, 1], 0),
        Err(NumaEvidenceError::CpuNodeMismatch {
            cpu: 1,
            expected_node: 0,
            observed_node: 1,
        })
    );
}

#[test]
fn memory_nodes_must_exist_in_evidence() {
    let evidence = NumaHostEvidence::new([(0, 0), (1, 1)]).unwrap();

    assert_eq!(evidence.validate_memory_node(0), Ok(()));
    assert_eq!(
        evidence.validate_memory_node(2),
        Err(NumaEvidenceError::UnknownMemoryNode(2))
    );
}

#[test]
fn malformed_topology_evidence_fails_closed() {
    assert_eq!(
        NumaHostEvidence::new([]),
        Err(NumaEvidenceError::EmptyTopology)
    );
    assert_eq!(
        NumaHostEvidence::new([(0, 0), (0, 1)]),
        Err(NumaEvidenceError::DuplicateCpu(0))
    );
}
