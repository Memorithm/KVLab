use core::fmt;

pub const BKV_K4_NUMA_PLAN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPlacement {
    LocalNode(u32),
    RemoteNode(u32),
    Interleave,
    ShardedByNode,
    ReplicatedReadMostly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaCase {
    pub name: String,
    pub worker_cpus: Vec<u32>,
    pub cpu_node: u32,
    pub memory: MemoryPlacement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NumaPlanError {
    EmptyCaseName,
    EmptyWorkerSet,
    DuplicateCpu(u32),
    NodeMismatch { cpu_node: u32, placement_node: u32 },
}

impl fmt::Display for NumaPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCaseName => write!(f, "BKV-K4 NUMA case name must be non-empty"),
            Self::EmptyWorkerSet => {
                write!(f, "BKV-K4 NUMA case must include at least one worker CPU")
            }
            Self::DuplicateCpu(cpu) => write!(f, "BKV-K4 NUMA case repeats logical CPU {cpu}"),
            Self::NodeMismatch {
                cpu_node,
                placement_node,
            } => write!(
                f,
                "local NUMA placement node {placement_node} does not match worker node {cpu_node}"
            ),
        }
    }
}

impl std::error::Error for NumaPlanError {}

impl NumaCase {
    pub fn new(
        name: impl Into<String>,
        worker_cpus: Vec<u32>,
        cpu_node: u32,
        memory: MemoryPlacement,
    ) -> Result<Self, NumaPlanError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(NumaPlanError::EmptyCaseName);
        }
        if worker_cpus.is_empty() {
            return Err(NumaPlanError::EmptyWorkerSet);
        }
        let mut sorted = worker_cpus.clone();
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            if pair[0] == pair[1] {
                return Err(NumaPlanError::DuplicateCpu(pair[0]));
            }
        }
        if let MemoryPlacement::LocalNode(placement_node) = memory {
            if placement_node != cpu_node {
                return Err(NumaPlanError::NodeMismatch {
                    cpu_node,
                    placement_node,
                });
            }
        }
        Ok(Self {
            name,
            worker_cpus,
            cpu_node,
            memory,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaCampaignPlan {
    pub schema_version: u32,
    pub host_evidence_sha256: String,
    pub commit_sha: String,
    pub cases: Vec<NumaCase>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NumaCampaignError {
    MissingHostEvidence,
    MissingCommitSha,
    EmptyCases,
    DuplicateCaseName(String),
}

impl NumaCampaignPlan {
    pub fn new(
        host_evidence_sha256: impl Into<String>,
        commit_sha: impl Into<String>,
        cases: Vec<NumaCase>,
    ) -> Result<Self, NumaCampaignError> {
        let host_evidence_sha256 = host_evidence_sha256.into();
        if host_evidence_sha256.trim().is_empty() {
            return Err(NumaCampaignError::MissingHostEvidence);
        }
        let commit_sha = commit_sha.into();
        if commit_sha.trim().is_empty() {
            return Err(NumaCampaignError::MissingCommitSha);
        }
        if cases.is_empty() {
            return Err(NumaCampaignError::EmptyCases);
        }
        let mut names: Vec<&str> = cases.iter().map(|case| case.name.as_str()).collect();
        names.sort_unstable();
        for pair in names.windows(2) {
            if pair[0] == pair[1] {
                return Err(NumaCampaignError::DuplicateCaseName(pair[0].to_owned()));
            }
        }
        Ok(Self {
            schema_version: BKV_K4_NUMA_PLAN_SCHEMA_VERSION,
            host_evidence_sha256,
            commit_sha,
            cases,
        })
    }
}
