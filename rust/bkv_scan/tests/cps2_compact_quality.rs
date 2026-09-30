use flat_attention_cps1::{AttentionShape, FlatAttentionConfig};
use kvlab_bkv_scan::cps2_compact_quality::{
    run_cps2_panel, Cps2Arm, Cps2Error, Cps2Protocol, CPS2_SCHEMA_VERSION, FLAT_CPS1_MERGE_REVISION,
};

fn shape(seq_len: usize, head_dim: usize) -> AttentionShape {
    AttentionShape {
        batch: 1,
        heads: 1,
        seq_len,
        head_dim,
    }
}

fn config(causal: bool) -> FlatAttentionConfig {
    FlatAttentionConfig {
        causal,
        softmax_scale: Some(1.0),
    }
}

fn synthetic_tensors(shape: AttentionShape) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = shape.tensor_len().unwrap();
    let q = (0..len)
        .map(|index| ((index * 7 % 19) as f32 - 9.0) / 11.0)
        .collect();
    let k = (0..len)
        .map(|index| ((index * 5 % 23) as f32 - 11.0) / 13.0)
        .collect();
    let v = (0..len)
        .map(|index| ((index * 3 % 29) as f32 - 14.0) / 17.0)
        .collect();
    (q, k, v)
}

fn row(
    panel: &kvlab_bkv_scan::cps2_compact_quality::Cps2Panel,
    arm: Cps2Arm,
    row: usize,
) -> &kvlab_bkv_scan::cps2_compact_quality::Cps2RowObservation {
    panel
        .observations
        .iter()
        .find(|observation| observation.arm == arm && observation.row == row)
        .unwrap()
}

#[test]
fn exact_producer_revision_and_schema_are_bound() {
    assert_eq!(CPS2_SCHEMA_VERSION, "kvlab.cps2-compact-quality/v1");
    assert_eq!(
        FLAT_CPS1_MERGE_REVISION,
        "ad1634fc922f6223dd3a83ac84154a82b1a35562"
    );
}

#[test]
fn all_accept_matches_dense_exactly_on_irregular_geometry() {
    let shape = AttentionShape {
        batch: 2,
        heads: 3,
        seq_len: 19,
        head_dim: 17,
    };
    let (q, k, v) = synthetic_tensors(shape);

    for causal in [false, true] {
        let panel = run_cps2_panel(
            &q,
            &k,
            &v,
            shape,
            config(causal),
            Cps2Protocol {
                compact_coordinates: &[0, 3, 16],
                candidate_budget: usize::MAX,
                reference_top_k: 4,
                random_seed: 7,
            },
        )
        .unwrap();
        for row_index in 0..shape.lse_len().unwrap() {
            let observation = row(&panel, Cps2Arm::AllAccept, row_index);
            assert_eq!(observation.output_max_abs_error, 0.0);
            assert_eq!(observation.lse_abs_error, 0.0);
            assert_eq!(observation.retained_softmax_mass, 1.0);
            assert_eq!(observation.omitted_softmax_mass, 0.0);
            assert_eq!(observation.selected_density, 1.0);
            assert_eq!(observation.top_k_recall, 1.0);
            assert!(!observation.timing_measured);
            assert!(!observation.physical_traffic_measured);
            assert!(!observation.model_quality_measured);
            assert!(!observation.promotion_authorized);
        }
    }
}

#[test]
fn matched_controls_use_exact_compact_density_for_every_causal_row() {
    let shape = shape(19, 17);
    let (q, k, v) = synthetic_tensors(shape);
    let panel = run_cps2_panel(
        &q,
        &k,
        &v,
        shape,
        config(true),
        Cps2Protocol {
            compact_coordinates: &[0, 3, 16],
            candidate_budget: 4,
            reference_top_k: 4,
            random_seed: 11,
        },
    )
    .unwrap();
    assert_eq!(panel.observations.len(), shape.lse_len().unwrap() * 5);

    for row_index in 0..shape.seq_len {
        let compact = row(&panel, Cps2Arm::CompactProjected, row_index);
        let recent = row(&panel, Cps2Arm::RecentTail, row_index);
        let random = row(&panel, Cps2Arm::MatchedRandom, row_index);
        let full = row(&panel, Cps2Arm::FullScoreTopK, row_index);
        let expected = 4.min(row_index + 1);

        assert_eq!(compact.selected_keys.len(), expected);
        assert_eq!(recent.selected_keys.len(), expected);
        assert_eq!(random.selected_keys.len(), expected);
        assert_eq!(full.selected_keys.len(), expected);
        assert_eq!(compact.selected_density, recent.selected_density);
        assert_eq!(compact.selected_density, random.selected_density);
        assert!(compact.selected_keys.iter().all(|&key| key <= row_index));
        assert!(recent.selected_keys.iter().all(|&key| key <= row_index));
        assert!(random.selected_keys.iter().all(|&key| key <= row_index));
    }
}

#[test]
fn full_score_topk_is_reference_ranking_upper_bound_when_budget_covers_k() {
    let shape = shape(19, 17);
    let (q, k, v) = synthetic_tensors(shape);
    let panel = run_cps2_panel(
        &q,
        &k,
        &v,
        shape,
        config(false),
        Cps2Protocol {
            compact_coordinates: &[0, 3, 16],
            candidate_budget: 8,
            reference_top_k: 4,
            random_seed: 13,
        },
    )
    .unwrap();

    for row_index in 0..shape.seq_len {
        let full = row(&panel, Cps2Arm::FullScoreTopK, row_index);
        assert_eq!(full.top_k_hits, 4);
        assert_eq!(full.top_k_recall, 1.0);
        assert!(full.retained_softmax_mass > 0.0);
        assert!(full.retained_softmax_mass <= 1.0);
    }
}

#[test]
fn omitted_dominant_coordinate_is_retained_as_negative_control() {
    let shape = shape(3, 2);
    let q = vec![1.0; 6];
    let k = vec![0.0, 20.0, 1.0, 0.0, 2.0, 0.0];
    let v = vec![10.0, 10.0, 0.0, 0.0, 0.0, 0.0];
    let panel = run_cps2_panel(
        &q,
        &k,
        &v,
        shape,
        config(false),
        Cps2Protocol {
            compact_coordinates: &[0],
            candidate_budget: 1,
            reference_top_k: 1,
            random_seed: 17,
        },
    )
    .unwrap();

    let compact = row(&panel, Cps2Arm::CompactProjected, 2);
    let full = row(&panel, Cps2Arm::FullScoreTopK, 2);

    assert_eq!(compact.selected_keys, vec![2]);
    assert_eq!(compact.reference_top_keys, vec![0]);
    assert_eq!(compact.top_k_recall, 0.0);
    assert!(compact.retained_softmax_mass < 1.0e-7);
    assert!(compact.output_max_abs_error > 9.9);

    assert_eq!(full.selected_keys, vec![0]);
    assert_eq!(full.top_k_recall, 1.0);
    assert!(full.retained_softmax_mass > 0.999_999);
    assert!(full.output_max_abs_error < 1.0e-5);
}

#[test]
fn random_control_is_seed_bound_and_repeatable() {
    let shape = shape(19, 5);
    let (q, k, v) = synthetic_tensors(shape);
    let first = run_cps2_panel(
        &q,
        &k,
        &v,
        shape,
        config(false),
        Cps2Protocol {
            compact_coordinates: &[0, 4],
            candidate_budget: 4,
            reference_top_k: 4,
            random_seed: 0x4350_5332,
        },
    )
    .unwrap();
    let second = run_cps2_panel(
        &q,
        &k,
        &v,
        shape,
        config(false),
        Cps2Protocol {
            compact_coordinates: &[0, 4],
            candidate_budget: 4,
            reference_top_k: 4,
            random_seed: 0x4350_5332,
        },
    )
    .unwrap();
    assert_eq!(first, second);

    let other = run_cps2_panel(
        &q,
        &k,
        &v,
        shape,
        config(false),
        Cps2Protocol {
            compact_coordinates: &[0, 4],
            candidate_budget: 4,
            reference_top_k: 4,
            random_seed: 0x4350_5333,
        },
    )
    .unwrap();
    let first_random = first
        .observations
        .iter()
        .filter(|observation| observation.arm == Cps2Arm::MatchedRandom)
        .map(|observation| observation.selected_keys.clone())
        .collect::<Vec<_>>();
    let other_random = other
        .observations
        .iter()
        .filter(|observation| observation.arm == Cps2Arm::MatchedRandom)
        .map(|observation| observation.selected_keys.clone())
        .collect::<Vec<_>>();
    assert_ne!(first_random, other_random);
}

#[test]
fn accounting_separates_compact_selector_from_full_score_control() {
    let shape = shape(19, 17);
    let (q, k, v) = synthetic_tensors(shape);
    let panel = run_cps2_panel(
        &q,
        &k,
        &v,
        shape,
        config(false),
        Cps2Protocol {
            compact_coordinates: &[0, 3, 16],
            candidate_budget: 4,
            reference_top_k: 4,
            random_seed: 19,
        },
    )
    .unwrap();

    assert_eq!(panel.compact_accounting.projection_dimension, 3);
    assert_eq!(panel.full_score_accounting.projection_dimension, 17);
    assert_eq!(
        panel.compact_accounting.evaluated_pairs,
        panel.full_score_accounting.evaluated_pairs
    );
    assert_eq!(
        panel.compact_accounting.evaluated_score_components,
        19 * 19 * 3
    );
    assert_eq!(
        panel.full_score_accounting.evaluated_score_components,
        19 * 19 * 17
    );
    assert_eq!(
        panel.compact_accounting.projected_key_payload_bytes,
        19 * 3 * core::mem::size_of::<f32>()
    );
    assert_eq!(
        panel.full_score_accounting.projected_key_payload_bytes,
        19 * 17 * core::mem::size_of::<f32>()
    );
}

#[test]
fn invalid_protocol_parameters_fail_closed() {
    let shape = shape(3, 2);
    let (q, k, v) = synthetic_tensors(shape);

    assert_eq!(
        run_cps2_panel(
            &q,
            &k,
            &v,
            shape,
            config(false),
            Cps2Protocol {
                compact_coordinates: &[0],
                candidate_budget: 0,
                reference_top_k: 1,
                random_seed: 1,
            }
        ),
        Err(Cps2Error::ZeroBudget)
    );
    assert_eq!(
        run_cps2_panel(
            &q,
            &k,
            &v,
            shape,
            config(false),
            Cps2Protocol {
                compact_coordinates: &[0],
                candidate_budget: 1,
                reference_top_k: 0,
                random_seed: 1,
            }
        ),
        Err(Cps2Error::ZeroReferenceTopK)
    );
    assert!(run_cps2_panel(
        &q,
        &k,
        &v,
        shape,
        config(false),
        Cps2Protocol {
            compact_coordinates: &[],
            candidate_budget: 1,
            reference_top_k: 1,
            random_seed: 1,
        }
    )
    .is_err());
}
