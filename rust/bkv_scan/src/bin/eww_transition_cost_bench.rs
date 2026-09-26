#![forbid(unsafe_code)]

//! EWW-K2 structural width-transition cost harness.
//!
//! This binary measures the generic ElasticWord v1 reference transition path
//! pinned by the KVLab preregistration. It uses lossless fixtures whose semantic
//! payload occupies lane 0 and whose higher lanes are zero, so every ordered
//! transition among the frozen widths is admissible under the reference
//! zero-extension/contraction oracle.
//!
//! Measurements are structural host measurements only. They do not establish
//! KV-cache, GPU, bandwidth, allocator, TTFT, TPOT or model-quality benefit.

use elastic_core::{ElasticWordPlaneV1, ElasticWordWidthV1, TransitionMechanism};
use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

const WIDTHS: [u16; 6] = [64, 128, 256, 512, 1024, 2048];
const SCHEMA: &str = "kvlab.eww-k2-transition-cost/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Config {
    words: usize,
    warmup: usize,
    repetitions: usize,
    seed: u64,
}

fn parse_usize(name: &str, value: Option<String>) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("missing {name}"))?
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn parse_u64(name: &str, value: Option<String>) -> Result<u64, String> {
    value
        .ok_or_else(|| format!("missing {name}"))?
        .parse::<u64>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn parse_config_from<I>(args: I) -> Result<Config, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let config = Config {
        words: parse_usize("words", args.next())?,
        warmup: parse_usize("warmup", args.next())?,
        repetitions: parse_usize("repetitions", args.next())?,
        seed: parse_u64("seed", args.next())?,
    };
    if args.next().is_some() {
        return Err("too many arguments".to_owned());
    }
    if config.words == 0 {
        return Err("words must be non-zero".to_owned());
    }
    if config.repetitions == 0 {
        return Err("repetitions must be non-zero".to_owned());
    }
    Ok(config)
}

fn parse_config() -> Result<Config, String> {
    parse_config_from(env::args().skip(1))
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn fixture(
    width: ElasticWordWidthV1,
    words: usize,
    seed: u64,
) -> Result<ElasticWordPlaneV1, String> {
    let lanes_per_word = usize::from(width.lanes());
    let lane_count = words
        .checked_mul(lanes_per_word)
        .ok_or_else(|| "fixture lane count overflow".to_owned())?;
    let mut lanes = vec![0_u64; lane_count];

    for word in 0..words {
        let word_u64 = u64::try_from(word).map_err(|_| "word index does not fit u64")?;
        lanes[word * lanes_per_word] = mix64(seed ^ word_u64);
    }

    ElasticWordPlaneV1::new(width, lanes).map_err(|error| error.to_string())
}

fn median_ns(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2
    } else {
        samples[middle]
    }
}

fn verify_roundtrip_payload(
    source: &ElasticWordPlaneV1,
    target: &ElasticWordPlaneV1,
) -> Result<(), String> {
    if source.word_count() != target.word_count() {
        return Err("word count changed during transition".to_owned());
    }

    for word_index in 0..source.word_count() {
        let source_word = source.word(word_index).map_err(|error| error.to_string())?;
        let target_word = target.word(word_index).map_err(|error| error.to_string())?;
        if source_word[0] != target_word[0] {
            return Err(format!("lane-0 payload drift at word {word_index}"));
        }
        if target_word.iter().skip(1).any(|value| *value != 0) {
            return Err(format!(
                "non-zero reserved target lane at word {word_index}"
            ));
        }
    }
    Ok(())
}

fn time_validation(
    source: &ElasticWordPlaneV1,
    target: ElasticWordWidthV1,
    repetitions: usize,
) -> Result<(u128, u128, u128), String> {
    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        let plan = source
            .plan_width_transition(target, TransitionMechanism::Reencode)
            .map_err(|error| error.to_string())?;
        black_box(plan);
        samples.push(start.elapsed().as_nanos());
    }
    let min = *samples.iter().min().expect("non-empty repetitions");
    let max = *samples.iter().max().expect("non-empty repetitions");
    let median = median_ns(&mut samples);
    Ok((median, min, max))
}

fn time_transition(
    source: &ElasticWordPlaneV1,
    target: ElasticWordWidthV1,
    repetitions: usize,
) -> Result<(ElasticWordPlaneV1, u128, u128, u128), String> {
    let mut samples = Vec::with_capacity(repetitions);
    let mut last = None;
    for _ in 0..repetitions {
        let start = Instant::now();
        let candidate = source
            .reference_repack_zero_extended(target)
            .map_err(|error| error.to_string())?;
        let elapsed = start.elapsed().as_nanos();
        black_box(candidate.as_lanes());
        last = Some(candidate);
        samples.push(elapsed);
    }
    let min = *samples.iter().min().expect("non-empty repetitions");
    let max = *samples.iter().max().expect("non-empty repetitions");
    let median = median_ns(&mut samples);
    Ok((last.expect("non-empty repetitions"), median, min, max))
}

fn time_verification(
    source: &ElasticWordPlaneV1,
    target: &ElasticWordPlaneV1,
    repetitions: usize,
) -> Result<(u128, u128, u128), String> {
    let mut samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let start = Instant::now();
        verify_roundtrip_payload(source, target)?;
        black_box(target.as_lanes());
        samples.push(start.elapsed().as_nanos());
    }
    let min = *samples.iter().min().expect("non-empty repetitions");
    let max = *samples.iter().max().expect("non-empty repetitions");
    let median = median_ns(&mut samples);
    Ok((median, min, max))
}

fn bytes_for(width: ElasticWordWidthV1, words: usize) -> Result<usize, String> {
    words
        .checked_mul(usize::from(width.bytes()))
        .ok_or_else(|| "byte count overflow".to_owned())
}

fn run(config: Config) -> Result<(), String> {
    println!(
        "schema,source_bits,target_bits,words,source_bytes,target_bytes,bytes_copied,bytes_initialized,bytes_discarded,bytes_rematerialized,validation_median_ns,validation_min_ns,validation_max_ns,transition_median_ns,transition_min_ns,transition_max_ns,verification_median_ns,verification_min_ns,verification_max_ns,rollback_median_ns,rollback_min_ns,rollback_max_ns"
    );

    for source_bits in WIDTHS {
        let source_width =
            ElasticWordWidthV1::from_bits(source_bits).map_err(|error| error.to_string())?;
        let source = fixture(source_width, config.words, config.seed)?;

        for target_bits in WIDTHS {
            if source_bits == target_bits {
                continue;
            }

            let target_width =
                ElasticWordWidthV1::from_bits(target_bits).map_err(|error| error.to_string())?;

            for _ in 0..config.warmup {
                let warm = source
                    .reference_repack_zero_extended(target_width)
                    .map_err(|error| error.to_string())?;
                verify_roundtrip_payload(&source, &warm)?;
                let rollback = warm
                    .reference_repack_zero_extended(source_width)
                    .map_err(|error| error.to_string())?;
                verify_roundtrip_payload(&source, &rollback)?;
                black_box(rollback.as_lanes());
            }

            let (validation_median, validation_min, validation_max) =
                time_validation(&source, target_width, config.repetitions)?;
            let (target, transition_median, transition_min, transition_max) =
                time_transition(&source, target_width, config.repetitions)?;
            let (verification_median, verification_min, verification_max) =
                time_verification(&source, &target, config.repetitions)?;
            let (_rollback, rollback_median, rollback_min, rollback_max) =
                time_transition(&target, source_width, config.repetitions)?;

            let source_bytes = bytes_for(source_width, config.words)?;
            let target_bytes = bytes_for(target_width, config.words)?;
            let common_lanes = usize::from(source_width.lanes().min(target_width.lanes()));
            let bytes_copied = config
                .words
                .checked_mul(common_lanes)
                .and_then(|value| value.checked_mul(8))
                .ok_or_else(|| "copied-byte count overflow".to_owned())?;
            let bytes_initialized = target_bytes.saturating_sub(source_bytes);
            let bytes_discarded = source_bytes.saturating_sub(target_bytes);

            println!(
                "{SCHEMA},{source_bits},{target_bits},{},{source_bytes},{target_bytes},{bytes_copied},{bytes_initialized},{bytes_discarded},0,{validation_median},{validation_min},{validation_max},{transition_median},{transition_min},{transition_max},{verification_median},{verification_min},{verification_max},{rollback_median},{rollback_min},{rollback_max}",
                config.words
            );
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    match parse_config().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!("usage: eww_transition_cost_bench <words> <warmup> <repetitions> <seed>");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ordered_frozen_width_transitions_are_lossless_for_fixture() {
        for source_bits in WIDTHS {
            let source_width = ElasticWordWidthV1::from_bits(source_bits).unwrap();
            let source = fixture(source_width, 7, 42).unwrap();

            for target_bits in WIDTHS {
                if source_bits == target_bits {
                    continue;
                }
                let target_width = ElasticWordWidthV1::from_bits(target_bits).unwrap();
                let target = source.reference_repack_zero_extended(target_width).unwrap();
                verify_roundtrip_payload(&source, &target).unwrap();

                let rollback = target.reference_repack_zero_extended(source_width).unwrap();
                assert_eq!(rollback, source);
            }
        }
    }

    #[test]
    fn byte_accounting_matches_width_delta() {
        let words = 5;
        let w64 = ElasticWordWidthV1::from_bits(64).unwrap();
        let w512 = ElasticWordWidthV1::from_bits(512).unwrap();

        assert_eq!(bytes_for(w64, words).unwrap(), 40);
        assert_eq!(bytes_for(w512, words).unwrap(), 320);

        let common = usize::from(w64.lanes().min(w512.lanes()));
        assert_eq!(words * common * 8, 40);
        assert_eq!(
            bytes_for(w512, words).unwrap() - bytes_for(w64, words).unwrap(),
            280
        );
    }

    #[test]
    fn parser_rejects_zero_words_or_repetitions() {
        assert!(parse_config_from([
            "0".to_owned(),
            "1".to_owned(),
            "3".to_owned(),
            "42".to_owned(),
        ])
        .unwrap_err()
        .contains("words"));
        assert!(parse_config_from([
            "10".to_owned(),
            "1".to_owned(),
            "0".to_owned(),
            "42".to_owned(),
        ])
        .unwrap_err()
        .contains("repetitions"));
    }
}
