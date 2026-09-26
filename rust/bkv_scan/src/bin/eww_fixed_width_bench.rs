#![forbid(unsafe_code)]

//! EWW-K1 fixed-width scalar benchmark harness.
//!
//! This binary measures the six frozen elastic-word widths on the same
//! deterministic nested workload. It emits raw timing/accounting records only;
//! it does not select a winner or make a performance, cache, bandwidth or
//! model-quality claim.

use kvlab_bkv_scan::scan_packed_pages_flat;
use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

const WIDTHS: [usize; 6] = [64, 128, 256, 512, 1024, 2048];
const BASIS_POINTS: usize = 10_000;
const SCHEMA: &str = "kvlab.eww-k1-fixed-width/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Config {
    pages: usize,
    warmup: usize,
    repetitions: usize,
    seed: u64,
    threshold_basis_points: usize,
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
        pages: parse_usize("pages", args.next())?,
        warmup: parse_usize("warmup", args.next())?,
        repetitions: parse_usize("repetitions", args.next())?,
        seed: parse_u64("seed", args.next())?,
        threshold_basis_points: parse_usize("threshold_basis_points", args.next())?,
    };

    if args.next().is_some() {
        return Err("too many arguments".to_owned());
    }
    if config.pages == 0 {
        return Err("pages must be non-zero".to_owned());
    }
    if config.repetitions == 0 {
        return Err("repetitions must be non-zero".to_owned());
    }
    if config.threshold_basis_points > BASIS_POINTS {
        return Err("threshold_basis_points must be in 0..=10000".to_owned());
    }
    Ok(config)
}

fn parse_config() -> Result<Config, String> {
    parse_config_from(env::args().skip(1))
}

/// Deterministic SplitMix64 finalizer.
///
/// Inputs are addressed by logical page and lane, so narrower widths are exact
/// prefixes of wider widths for the same page rather than unrelated PRNG runs.
fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn lane_value(seed: u64, page_id: usize, lane: usize) -> u64 {
    let page = u64::try_from(page_id).unwrap_or(u64::MAX);
    let lane = u64::try_from(lane).unwrap_or(u64::MAX);
    mix64(
        seed ^ page.wrapping_mul(0xd6e8_feb8_6659_fd93) ^ lane.wrapping_mul(0xa076_1d64_78bd_642f),
    )
}

fn build_workload(
    pages: usize,
    signature_bits: usize,
    seed: u64,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let words = signature_bits
        .checked_div(64)
        .ok_or_else(|| "invalid signature width".to_owned())?;
    if words == 0 || words * 64 != signature_bits {
        return Err("EWW-K1 widths must be non-zero multiples of 64".to_owned());
    }

    let total_words = pages
        .checked_mul(words)
        .ok_or_else(|| "flat plane word count overflow".to_owned())?;

    let query = (0..words)
        .map(|lane| lane_value(seed ^ 0x243f_6a88_85a3_08d3, usize::MAX / 2, lane))
        .collect::<Vec<_>>();

    let mut flat_pages = Vec::with_capacity(total_words);
    for page_id in 0..pages {
        for lane in 0..words {
            flat_pages.push(lane_value(seed, page_id, lane));
        }
    }

    Ok((query, flat_pages))
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

fn threshold_bits(signature_bits: usize, basis_points: usize) -> Result<usize, String> {
    signature_bits
        .checked_mul(basis_points)
        .map(|scaled| scaled / BASIS_POINTS)
        .ok_or_else(|| "threshold calculation overflow".to_owned())
}

fn run(config: Config) -> Result<(), String> {
    println!(
        "schema,width_bits,lanes_u64,pages,plane_bytes,threshold_basis_points,threshold_bits,warmup,repetitions,seed,selected_pages,bits_compared,median_ns,min_ns,max_ns"
    );

    for signature_bits in WIDTHS {
        let (query, flat_pages) = build_workload(config.pages, signature_bits, config.seed)?;
        let max_distance = threshold_bits(signature_bits, config.threshold_basis_points)?;
        let plane_bytes = flat_pages
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| "plane byte count overflow".to_owned())?;

        let reference = scan_packed_pages_flat(
            signature_bits,
            &query,
            config.pages,
            &flat_pages,
            max_distance,
        )
        .map_err(|error| format!("reference scan failed at W{signature_bits}: {error:?}"))?;

        for _ in 0..config.warmup {
            black_box(
                scan_packed_pages_flat(
                    signature_bits,
                    &query,
                    config.pages,
                    &flat_pages,
                    max_distance,
                )
                .map_err(|error| format!("warmup failed at W{signature_bits}: {error:?}"))?,
            );
        }

        let mut samples = Vec::with_capacity(config.repetitions);
        for _ in 0..config.repetitions {
            let start = Instant::now();
            let result = scan_packed_pages_flat(
                signature_bits,
                &query,
                config.pages,
                &flat_pages,
                max_distance,
            )
            .map_err(|error| format!("measurement failed at W{signature_bits}: {error:?}"))?;

            if result.selected_pages != reference.selected_pages {
                return Err(format!(
                    "candidate-set drift across repetitions at W{signature_bits}"
                ));
            }
            black_box(result);
            samples.push(start.elapsed().as_nanos());
        }

        let min_ns = *samples.iter().min().expect("non-empty repetitions");
        let max_ns = *samples.iter().max().expect("non-empty repetitions");
        let median = median_ns(&mut samples);

        println!(
            "{SCHEMA},{signature_bits},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            signature_bits / 64,
            config.pages,
            plane_bytes,
            config.threshold_basis_points,
            max_distance,
            config.warmup,
            config.repetitions,
            config.seed,
            reference.selected_pages.len(),
            reference.bits_compared,
            median,
            min_ns,
            max_ns
        );
    }

    Ok(())
}

fn main() -> ExitCode {
    match parse_config().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!(
                "usage: eww_fixed_width_bench <pages> <warmup> <repetitions> <seed> <threshold_basis_points>"
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_out_of_range_threshold() {
        let error = parse_config_from([
            "16".to_owned(),
            "1".to_owned(),
            "3".to_owned(),
            "42".to_owned(),
            "10001".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("0..=10000"));
    }

    #[test]
    fn wider_workloads_preserve_narrow_page_prefixes() {
        let (_, w64) = build_workload(3, 64, 7).unwrap();
        let (_, w256) = build_workload(3, 256, 7).unwrap();

        for page_id in 0..3 {
            assert_eq!(w64[page_id], w256[page_id * 4]);
        }
    }

    #[test]
    fn frozen_widths_execute_with_raw_accounting() {
        for bits in WIDTHS {
            let (query, pages) = build_workload(4, bits, 9).unwrap();
            let result = scan_packed_pages_flat(bits, &query, 4, &pages, bits / 2).unwrap();
            assert_eq!(result.pages_scanned, 4);
            assert_eq!(result.signature_bits, bits);
            assert_eq!(result.bits_compared, 4 * bits);
        }
    }
}
