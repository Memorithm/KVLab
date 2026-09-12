#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::process::ExitCode;

const BENCH_SCHEMA: &str = "bkv-k3-v1";
const SUMMARY_SCHEMA: &str = "bkv-k3-scaling-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
struct BenchSample {
    seed: u64,
    pages: usize,
    signature_bits: usize,
    max_distance: usize,
    workers: usize,
    warmup: usize,
    repetitions: usize,
    selected_pages: usize,
    bits_compared: usize,
    median_ns: u128,
    min_ns: u128,
    max_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
struct ScalingPoint {
    workers: usize,
    median_ns: u128,
    speedup: f64,
    efficiency: f64,
}

fn parse_field<T>(name: &str, value: Option<&str>) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .ok_or_else(|| format!("missing {name}"))?
        .parse::<T>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn parse_sample(line: &str) -> Result<BenchSample, String> {
    let mut fields = line.split(',');
    let schema = fields.next().ok_or_else(|| "missing schema".to_owned())?;
    if schema != BENCH_SCHEMA {
        return Err(format!("unsupported schema: {schema}"));
    }

    let sample = BenchSample {
        seed: parse_field("seed", fields.next())?,
        pages: parse_field("pages", fields.next())?,
        signature_bits: parse_field("signature_bits", fields.next())?,
        max_distance: parse_field("max_distance", fields.next())?,
        workers: parse_field("workers", fields.next())?,
        warmup: parse_field("warmup", fields.next())?,
        repetitions: parse_field("repetitions", fields.next())?,
        selected_pages: parse_field("selected_pages", fields.next())?,
        bits_compared: parse_field("bits_compared", fields.next())?,
        median_ns: parse_field("median_ns", fields.next())?,
        min_ns: parse_field("min_ns", fields.next())?,
        max_ns: parse_field("max_ns", fields.next())?,
    };
    if fields.next().is_some() {
        return Err("too many CSV fields".to_owned());
    }
    if sample.workers == 0 || sample.pages == 0 || sample.signature_bits == 0 {
        return Err("workers, pages, and signature_bits must be non-zero".to_owned());
    }
    if sample.repetitions == 0 || sample.median_ns == 0 {
        return Err("repetitions and median_ns must be non-zero".to_owned());
    }
    if sample.min_ns > sample.median_ns || sample.median_ns > sample.max_ns {
        return Err("timing order must satisfy min <= median <= max".to_owned());
    }
    Ok(sample)
}

fn same_campaign(left: &BenchSample, right: &BenchSample) -> bool {
    left.seed == right.seed
        && left.pages == right.pages
        && left.signature_bits == right.signature_bits
        && left.max_distance == right.max_distance
        && left.warmup == right.warmup
        && left.repetitions == right.repetitions
        && left.selected_pages == right.selected_pages
        && left.bits_compared == right.bits_compared
}

fn summarize(samples: &[BenchSample]) -> Result<Vec<ScalingPoint>, String> {
    if samples.is_empty() {
        return Err("no benchmark samples supplied".to_owned());
    }

    let reference = &samples[0];
    if samples.iter().any(|sample| !same_campaign(reference, sample)) {
        return Err(
            "samples do not describe the same deterministic benchmark campaign".to_owned(),
        );
    }

    let mut ordered = samples.to_vec();
    ordered.sort_unstable_by_key(|sample| sample.workers);
    for pair in ordered.windows(2) {
        if pair[0].workers == pair[1].workers {
            return Err(format!("duplicate worker count: {}", pair[0].workers));
        }
    }

    let baseline = ordered
        .iter()
        .find(|sample| sample.workers == 1)
        .ok_or_else(|| "missing workers=1 baseline".to_owned())?;
    let baseline_ns = baseline.median_ns as f64;

    Ok(ordered
        .into_iter()
        .map(|sample| {
            let speedup = baseline_ns / sample.median_ns as f64;
            ScalingPoint {
                workers: sample.workers,
                median_ns: sample.median_ns,
                speedup,
                efficiency: speedup / sample.workers as f64,
            }
        })
        .collect())
}

fn read_samples(paths: &[String]) -> Result<Vec<BenchSample>, String> {
    if paths.is_empty() {
        return Err("at least one benchmark CSV path is required".to_owned());
    }

    let mut samples = Vec::new();
    for path in paths {
        let text = fs::read_to_string(path).map_err(|error| format!("{path}: {error}"))?;
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("schema,") {
                continue;
            }
            samples.push(
                parse_sample(line)
                    .map_err(|error| format!("{path}: line {}: {error}", index + 1))?,
            );
        }
    }
    Ok(samples)
}

fn run(paths: &[String]) -> Result<(), String> {
    let samples = read_samples(paths)?;
    let points = summarize(&samples)?;
    println!("schema,workers,median_ns,speedup,efficiency");
    for point in points {
        println!(
            "{SUMMARY_SCHEMA},{},{},{:.6},{:.6}",
            point.workers, point.median_ns, point.speedup, point.efficiency
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let paths = env::args().skip(1).collect::<Vec<_>>();
    match run(&paths) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!("usage: bkv_scaling_summary <bkv_scan_bench.csv> [more.csv ...]");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(workers: usize, median_ns: u128) -> BenchSample {
        BenchSample {
            seed: 7,
            pages: 100_000,
            signature_bits: 256,
            max_distance: 96,
            workers,
            warmup: 5,
            repetitions: 25,
            selected_pages: 14_453,
            bits_compared: 25_600_000,
            median_ns,
            min_ns: median_ns - 10,
            max_ns: median_ns + 10,
        }
    }

    #[test]
    fn parses_current_benchmark_schema() {
        let line = "bkv-k3-v1,7,100000,256,96,4,5,25,14453,25600000,300,290,310";
        assert_eq!(parse_sample(line).unwrap(), sample(4, 300));
    }

    #[test]
    fn computes_speedup_and_efficiency_from_single_worker_baseline() {
        let points = summarize(&[sample(4, 250), sample(1, 1_000), sample(2, 500)]).unwrap();
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].workers, 1);
        assert_eq!(points[0].speedup, 1.0);
        assert_eq!(points[1].speedup, 2.0);
        assert_eq!(points[1].efficiency, 1.0);
        assert_eq!(points[2].speedup, 4.0);
        assert_eq!(points[2].efficiency, 1.0);
    }

    #[test]
    fn rejects_campaign_drift() {
        let mut drifted = sample(2, 500);
        drifted.seed = 8;
        assert!(summarize(&[sample(1, 1_000), drifted])
            .unwrap_err()
            .contains("same deterministic benchmark campaign"));
    }

    #[test]
    fn rejects_candidate_count_drift() {
        let mut drifted = sample(2, 500);
        drifted.selected_pages += 1;
        assert!(summarize(&[sample(1, 1_000), drifted])
            .unwrap_err()
            .contains("same deterministic benchmark campaign"));
    }

    #[test]
    fn rejects_duplicate_workers() {
        assert_eq!(
            summarize(&[sample(1, 1_000), sample(1, 900)]).unwrap_err(),
            "duplicate worker count: 1"
        );
    }

    #[test]
    fn rejects_missing_single_worker_baseline() {
        assert_eq!(
            summarize(&[sample(2, 500), sample(4, 250)]).unwrap_err(),
            "missing workers=1 baseline"
        );
    }
}
