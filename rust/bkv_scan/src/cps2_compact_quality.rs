//! CPS-2 comparative quality evaluator for FLAT compact preselection.
//!
//! KVLab consumes the exact FLAT CPS-1 producer revision and evaluates candidate
//! sets against dense/full-fidelity attention. This module is research-only:
//! selector work counts and selected numerical pairs are logical accounting, not
//! latency, physical traffic, bandwidth, memory-residency or model-quality claims.

use core::fmt;

use flat_algebraic_attention::compact_preselection::{compact_preselect, CompactPreselectionError};
use flat_attention_cps1::api::research_structural_routing::{
    forward_reference_structural_sparse, StructuralCandidateSet, StructuralRoutingError,
};
use flat_attention_cps1::{forward_reference, AttentionShape, FlatAttentionConfig, FlatAttentionError};

pub const CPS2_SCHEMA_VERSION: &str = "kvlab.cps2-compact-quality/v1";
pub const FLAT_CPS1_MERGE_REVISION: &str = "ad1634fc922f6223dd3a83ac84154a82b1a35562";
pub const MATCHED_RANDOM_ALGORITHM: &str = "splitmix64-row-page-ranking-v1";
pub const RECENCY_ALGORITHM: &str = "tail-window-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cps2Arm {
    AllAccept,
    CompactProjected,
    FullScoreTopK,
    RecentTail,
    MatchedRandom,
}

impl Cps2Arm {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::AllAccept => "all_accept",
            Self::CompactProjected => "compact_projected",
            Self::FullScoreTopK => "full_score_topk",
            Self::RecentTail => "recent_tail",
            Self::MatchedRandom => "matched_random",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cps2SelectorAccounting {
    pub projection_dimension: usize,
    pub projected_key_payload_bytes: usize,
    pub evaluated_pairs: usize,
    pub evaluated_score_components: usize,
    pub selected_pairs: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cps2RowObservation {
    pub schema_version: &'static str,
    pub flat_source_revision: &'static str,
    pub arm: Cps2Arm,
    pub row: usize,
    pub eligible_keys: usize,
    pub selected_keys: Vec<usize>,
    pub reference_top_keys: Vec<usize>,
    pub top_k_hits: usize,
    pub top_k_recall: f64,
    pub retained_softmax_mass: f64,
    pub omitted_softmax_mass: f64,
    pub selected_density: f64,
    pub selector_score_components: usize,
    pub numerical_pairs_executed: usize,
    pub output_max_abs_error: f64,
    pub lse_abs_error: f64,
    pub timing_measured: bool,
    pub physical_traffic_measured: bool,
    pub model_quality_measured: bool,
    pub promotion_authorized: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cps2Panel {
    pub schema_version: &'static str,
    pub flat_source_revision: &'static str,
    pub compact_accounting: Cps2SelectorAccounting,
    pub full_score_accounting: Cps2SelectorAccounting,
    pub observations: Vec<Cps2RowObservation>,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Cps2Error {
    Attention(FlatAttentionError),
    Compact(CompactPreselectionError),
    Structural(StructuralRoutingError),
    ZeroBudget,
    ZeroReferenceTopK,
    NonFiniteReferenceScore { row: usize, key_position: usize },
    NonFiniteAttentionOutput { arm: Cps2Arm, row: usize },
}

impl fmt::Display for Cps2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attention(error) => write!(f, "attention input failed validation: {error}"),
            Self::Compact(error) => write!(f, "compact selector failed validation: {error}"),
            Self::Structural(error) => write!(f, "structural attention failed validation: {error}"),
            Self::ZeroBudget => write!(f, "CPS-2 candidate budget must be non-zero"),
            Self::ZeroReferenceTopK => write!(f, "CPS-2 reference top-k must be non-zero"),
            Self::NonFiniteReferenceScore { row, key_position } => write!(
                f,
                "reference score is non-finite for row {row}, key {key_position}"
            ),
            Self::NonFiniteAttentionOutput { arm, row } => write!(
                f,
                "attention output is non-finite for arm {}, row {row}",
                arm.label()
            ),
        }
    }
}

impl std::error::Error for Cps2Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Attention(error) => Some(error),
            Self::Compact(error) => Some(error),
            Self::Structural(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FlatAttentionError> for Cps2Error {
    fn from(value: FlatAttentionError) -> Self {
        Self::Attention(value)
    }
}

impl From<CompactPreselectionError> for Cps2Error {
    fn from(value: CompactPreselectionError) -> Self {
        Self::Compact(value)
    }
}

impl From<StructuralRoutingError> for Cps2Error {
    fn from(value: StructuralRoutingError) -> Self {
        Self::Structural(value)
    }
}

pub fn run_cps2_panel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: AttentionShape,
    config: FlatAttentionConfig,
    compact_coordinates: &[usize],
    candidate_budget: usize,
    reference_top_k: usize,
    random_seed: u64,
) -> Result<Cps2Panel, Cps2Error> {
    if candidate_budget == 0 {
        return Err(Cps2Error::ZeroBudget);
    }
    if reference_top_k == 0 {
        return Err(Cps2Error::ZeroReferenceTopK);
    }

    let dense = forward_reference(q, k, v, shape, config)?;
    validate_output(Cps2Arm::AllAccept, shape, &dense.output, &dense.lse)?;

    let compact = compact_preselect(q, k, shape, config, compact_coordinates, candidate_budget)?;
    let full_coordinates = (0..shape.head_dim).collect::<Vec<_>>();
    let full_score = compact_preselect(q, k, shape, config, &full_coordinates, candidate_budget)?;

    let query_rows = shape.lse_len()?;
    let mut all_rows = Vec::with_capacity(query_rows);
    let mut recent_rows = Vec::with_capacity(query_rows);
    let mut random_rows = Vec::with_capacity(query_rows);

    for row in 0..query_rows {
        let eligible = eligible_key_count(shape, config, row);
        let selected_count = compact.candidates.row(row)?.len();

        all_rows.push((0..eligible).collect::<Vec<_>>());
        recent_rows.push((eligible - selected_count..eligible).collect::<Vec<_>>());

        let mut ranked = (0..eligible)
            .map(|key| (random_rank(random_seed, row, key), key))
            .collect::<Vec<_>>();
        ranked.sort_unstable();
        let mut selected = ranked
            .into_iter()
            .take(selected_count)
            .map(|(_, key)| key)
            .collect::<Vec<_>>();
        selected.sort_unstable();
        random_rows.push(selected);
    }

    let all_accept = StructuralCandidateSet::from_rows(shape, all_rows)?;
    let recent = StructuralCandidateSet::from_rows(shape, recent_rows)?;
    let random = StructuralCandidateSet::from_rows(shape, random_rows)?;

    let arms = vec![
        (Cps2Arm::AllAccept, all_accept),
        (Cps2Arm::CompactProjected, compact.candidates.clone()),
        (Cps2Arm::FullScoreTopK, full_score.candidates.clone()),
        (Cps2Arm::RecentTail, recent),
        (Cps2Arm::MatchedRandom, random),
    ];

    let scale = f64::from(config.resolved_scale(shape.head_dim)?);
    let mut observations = Vec::with_capacity(query_rows * arms.len());

    for (arm, candidates) in arms {
        let sparse = forward_reference_structural_sparse(q, k, v, shape, config, &candidates)?;
        validate_output(arm, shape, &sparse.attention.output, &sparse.attention.lse)?;

        for row in 0..query_rows {
            let eligible = eligible_key_count(shape, config, row);
            let scores = reference_scores(q, k, shape, row, eligible, scale)?;
            let selected = candidates.row(row)?.to_vec();
            let reference_top = reference_top_keys(&scores, reference_top_k.min(eligible));
            let hits = reference_top
                .iter()
                .filter(|key| selected.binary_search(key).is_ok())
                .count();
            let top_k_recall = hits as f64 / reference_top.len() as f64;
            let retained = retained_mass(&scores, &selected);
            let selected_density = selected.len() as f64 / eligible as f64;
            let selector_score_components = match arm {
                Cps2Arm::CompactProjected => eligible * compact_coordinates.len(),
                Cps2Arm::FullScoreTopK => eligible * shape.head_dim,
                Cps2Arm::AllAccept | Cps2Arm::RecentTail | Cps2Arm::MatchedRandom => 0,
            };

            let output_begin = row * shape.head_dim;
            let output_end = output_begin + shape.head_dim;
            let output_max_abs_error = dense.output[output_begin..output_end]
                .iter()
                .zip(&sparse.attention.output[output_begin..output_end])
                .map(|(&reference, &candidate)| f64::from((reference - candidate).abs()))
                .fold(0.0_f64, f64::max);
            let lse_abs_error = f64::from((dense.lse[row] - sparse.attention.lse[row]).abs());

            observations.push(Cps2RowObservation {
                schema_version: CPS2_SCHEMA_VERSION,
                flat_source_revision: FLAT_CPS1_MERGE_REVISION,
                arm,
                row,
                eligible_keys: eligible,
                selected_keys: selected,
                reference_top_keys: reference_top,
                top_k_hits: hits,
                top_k_recall,
                retained_softmax_mass: retained,
                omitted_softmax_mass: 1.0 - retained,
                selected_density,
                selector_score_components,
                numerical_pairs_executed: candidates.row(row)?.len(),
                output_max_abs_error,
                lse_abs_error,
                timing_measured: false,
                physical_traffic_measured: false,
                model_quality_measured: false,
                promotion_authorized: false,
            });
        }
    }

    Ok(Cps2Panel {
        schema_version: CPS2_SCHEMA_VERSION,
        flat_source_revision: FLAT_CPS1_MERGE_REVISION,
        compact_accounting: Cps2SelectorAccounting {
            projection_dimension: compact.counters.projection_dimension,
            projected_key_payload_bytes: compact.counters.projected_key_payload_bytes,
            evaluated_pairs: compact.counters.evaluated_pairs,
            evaluated_score_components: compact.counters.evaluated_score_components,
            selected_pairs: compact.counters.selected_pairs,
        },
        full_score_accounting: Cps2SelectorAccounting {
            projection_dimension: full_score.counters.projection_dimension,
            projected_key_payload_bytes: full_score.counters.projected_key_payload_bytes,
            evaluated_pairs: full_score.counters.evaluated_pairs,
            evaluated_score_components: full_score.counters.evaluated_score_components,
            selected_pairs: full_score.counters.selected_pairs,
        },
        observations,
    })
}

fn eligible_key_count(shape: AttentionShape, config: FlatAttentionConfig, row: usize) -> usize {
    if config.causal {
        row % shape.seq_len + 1
    } else {
        shape.seq_len
    }
}

fn reference_scores(
    q: &[f32],
    k: &[f32],
    shape: AttentionShape,
    row: usize,
    eligible: usize,
    scale: f64,
) -> Result<Vec<f64>, Cps2Error> {
    let query = &q[row * shape.head_dim..(row + 1) * shape.head_dim];
    let head_index = row / shape.seq_len;
    let head_base = head_index * shape.seq_len * shape.head_dim;
    let mut scores = Vec::with_capacity(eligible);
    for key_position in 0..eligible {
        let key_base = head_base + key_position * shape.head_dim;
        let score = query
            .iter()
            .zip(&k[key_base..key_base + shape.head_dim])
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>()
            * scale;
        if !score.is_finite() {
            return Err(Cps2Error::NonFiniteReferenceScore { row, key_position });
        }
        scores.push(if score == 0.0 { 0.0 } else { score });
    }
    Ok(scores)
}

fn reference_top_keys(scores: &[f64], count: usize) -> Vec<usize> {
    let mut ranked = (0..scores.len()).collect::<Vec<_>>();
    ranked.sort_by(|&left, &right| {
        scores[right]
            .total_cmp(&scores[left])
            .then_with(|| left.cmp(&right))
    });
    ranked.truncate(count);
    ranked.sort_unstable();
    ranked
}

fn retained_mass(scores: &[f64], selected: &[usize]) -> f64 {
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights = scores
        .iter()
        .map(|score| (*score - max_score).exp())
        .collect::<Vec<_>>();
    let partition = weights.iter().sum::<f64>();
    let selected_partition = selected.iter().map(|&key| weights[key]).sum::<f64>();
    (selected_partition / partition).clamp(0.0, 1.0)
}

fn validate_output(
    arm: Cps2Arm,
    shape: AttentionShape,
    output: &[f32],
    lse: &[f32],
) -> Result<(), Cps2Error> {
    for row in 0..shape.lse_len()? {
        let begin = row * shape.head_dim;
        let end = begin + shape.head_dim;
        if output[begin..end].iter().any(|value| !value.is_finite()) || !lse[row].is_finite() {
            return Err(Cps2Error::NonFiniteAttentionOutput { arm, row });
        }
    }
    Ok(())
}

fn random_rank(seed: u64, row: usize, key: usize) -> u64 {
    let row_seed = splitmix64(seed ^ row as u64);
    splitmix64(row_seed ^ key as u64)
}

fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_rank_is_repeatable_and_row_scoped() {
        assert_eq!(random_rank(7, 3, 11), random_rank(7, 3, 11));
        assert_ne!(random_rank(7, 3, 11), random_rank(7, 4, 11));
        assert_ne!(random_rank(7, 3, 11), random_rank(8, 3, 11));
    }

    #[test]
    fn reference_ties_choose_smallest_original_key() {
        assert_eq!(reference_top_keys(&[0.0, 0.0, 0.0, 0.0], 2), vec![0, 1]);
    }
}
