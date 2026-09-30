use kvlab_bkv_scan::cps2_compact_quality::{run_cps2_panel, Cps2Arm, Cps2RowObservation};
use flat_attention::{AttentionShape, FlatAttentionConfig};

fn ids(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("|")
}

fn emit(case: &str, observation: &Cps2RowObservation) {
    println!(
        "{},{},{},{},{},{},{:.12},{:.12},{:.9},{:.9},{},{},{},{},{},{},{}",
        observation.schema_version,
        observation.flat_source_revision,
        case,
        observation.arm.label(),
        observation.row,
        ids(&observation.selected_keys),
        observation.top_k_recall,
        observation.retained_softmax_mass,
        observation.output_max_abs_error,
        observation.lse_abs_error,
        observation.eligible_keys,
        observation.selector_score_components,
        observation.numerical_pairs_executed,
        observation.timing_measured,
        observation.physical_traffic_measured,
        observation.model_quality_measured,
        observation.promotion_authorized
    );
}

fn run_case(adverse: bool) -> Result<(), Box<dyn std::error::Error>> {
    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len: 5,
        head_dim: 2,
    };
    let config = FlatAttentionConfig {
        causal: false,
        softmax_scale: Some(1.0),
    };
    let q = [1.0, if adverse { 1.0 } else { 0.0 }].repeat(5);
    let mut k = Vec::with_capacity(10);
    let mut v = Vec::with_capacity(10);
    for key in 0..5 {
        k.extend_from_slice(&[key as f32, if adverse && key == 0 { 20.0 } else { 0.0 }]);
        v.extend_from_slice(&[key as f32, 4.0 - key as f32]);
    }

    let panel = run_cps2_panel(&q, &k, &v, shape, config, &[0], 2, 2, 0x4350_5332)?;
    let case = if adverse {
        "omitted_dominant_coordinate"
    } else {
        "aligned_coordinate"
    };
    for arm in [
        Cps2Arm::AllAccept,
        Cps2Arm::CompactProjected,
        Cps2Arm::FullScoreTopK,
        Cps2Arm::RecentTail,
        Cps2Arm::MatchedRandom,
    ] {
        let observation = panel
            .observations
            .iter()
            .find(|observation| observation.arm == arm && observation.row == 4)
            .expect("frozen row must exist");
        emit(case, observation);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "schema,flat_source_revision,case,arm,row,selected_ids,top_k_recall,retained_softmax_mass,output_max_abs_error,lse_abs_error,eligible_keys,selector_score_components,numerical_pairs_executed,timing_measured,physical_traffic_measured,model_quality_measured,promotion_authorized"
    );
    run_case(false)?;
    run_case(true)?;
    Ok(())
}
