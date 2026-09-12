use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaHostEvidence {
    cpu_to_node: BTreeMap<u32, u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NumaEvidenceError {
    EmptyTopology,
    DuplicateCpu(u32),
    UnknownCpu(u32),
    CpuNodeMismatch {
        cpu: u32,
        expected_node: u32,
        observed_node: u32,
    },
    UnknownMemoryNode(u32),
}

impl fmt::Display for NumaEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTopology => write!(f, "NUMA host evidence must contain at least one logical CPU"),
            Self::DuplicateCpu(cpu) => write!(f, "NUMA host evidence repeats logical CPU {cpu}"),
            Self::UnknownCpu(cpu) => write!(f, "NUMA plan references unknown logical CPU {cpu}"),
            Self::CpuNodeMismatch {
                cpu,
                expected_node,
                observed_node,
            } => write!(
                f,
                "logical CPU {cpu} is on NUMA node {observed_node}, not declared worker node {expected_node}"
            ),
            Self::UnknownMemoryNode(node) => {
                write!(f, "NUMA plan references unknown memory node {node}")
            }
        }
    }
}

impl std::error::Error for NumaEvidenceError {}

impl NumaHostEvidence {
    pub fn new(entries: impl IntoIterator<Item = (u32, u32)>) -> Result<Self, NumaEvidenceError> {
        let mut cpu_to_node = BTreeMap::new();
        for (cpu, node) in entries {
            if cpu_to_node.insert(cpu, node).is_some() {
                return Err(NumaEvidenceError::DuplicateCpu(cpu));
            }
        }
        if cpu_to_node.is_empty() {
            return Err(NumaEvidenceError::EmptyTopology);
        }
        Ok(Self { cpu_to_node })
    }

    #[must_use]
    pub fn node_for_cpu(&self, cpu: u32) -> Option<u32> {
        self.cpu_to_node.get(&cpu).copied()
    }

    #[must_use]
    pub fn contains_node(&self, node: u32) -> bool {
        self.cpu_to_node.values().any(|candidate| *candidate == node)
    }

    pub fn validate_worker_set(
        &self,
        worker_cpus: &[u32],
        declared_node: u32,
    ) -> Result<(), NumaEvidenceError> {
        for &cpu in worker_cpus {
            let observed_node = self.node_for_cpu(cpu).ok_or(NumaEvidenceError::UnknownCpu(cpu))?;
            if observed_node != declared_node {
                return Err(NumaEvidenceError::CpuNodeMismatch {
                    cpu,
                    expected_node: declared_node,
                    observed_node,
                });
            }
        }
        Ok(())
    }

    pub fn validate_memory_node(&self, node: u32) -> Result<(), NumaEvidenceError> {
        if self.contains_node(node) {
            Ok(())
        } else {
            Err(NumaEvidenceError::UnknownMemoryNode(node))
        }
    }
}
