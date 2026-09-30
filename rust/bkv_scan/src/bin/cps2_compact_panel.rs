use flat_attention_cps1::{AttentionShape, FlatAttentionConfig};
use kvlab_bkv_scan::cps2_compact_quality::{run_cps2_panel, Cps2Arm, Cps2RowObservation};

fn ids(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("|")
}

fn emit(case: &str, observation: &Cps2RowObservation) {
    println!(
        "{},{},{},{},{},{},{},{},{},{:.12},{:.12},{:.12},{:.12},{},{},{:.9},{:.9},{},{},{},{}",
        observation.schema_version,
        observation.flat_source_revision,
        case,
        observation.arm.label(),
        observation.row,
        observation.eligible_keys,
        ids(&observation.selected_keys),
        ids(&observation.reference_top_keys),
        observation.top_k_hits,
        observation.top_k_recall,
        observation.retained_softmax_mass,
        observation.omitted_softmax_mass,
        observation.selected_density,
        observation.selector_score_components,
        observation.numerical_pairs_executed,
        observation.output_max_abs_error,
        observation.lse_abs_error,
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
    for observation in &panel.observations {
        emit(case, observation);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "schema,flat_source_revision,case,arm,row,eligible_keys,selected_ids,reference_top_ids,top_k_hits,top_k_recall,retained_softmax_mass,omitted_softmax_mass,selected_density,selector_score_components,numerical_pairs_executed,output_max_abs_error,lse_abs_error,timing_measured,physical_traffic_measured,model_quality_measured,promotion_authorized"
    );
    run_case(false)?;
    run_case(true)?;
    Ok(())
}
