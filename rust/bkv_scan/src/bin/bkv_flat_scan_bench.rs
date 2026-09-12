#![forbid(unsafe_code)]

use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

use kvlab_bkv_scan::scan_packed_pages;

#[derive(Debug, Clone, Copy)]
struct Config {
    pages: usize,
    signature_bits: usize,
    max_distance: usize,
    workers: usize,
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

fn parse_config() -> Result<Config, String> {
    let mut args = env::args().skip(1);
    let config = Config {
        pages: parse_usize("pages", args.next())?,
        signature_bits: parse_usize("signature_bits", args.next())?,
        max_distance: parse_usize("max_distance", args.next())?,
        workers: parse_usize("workers", args.next())?,
        warmup: parse_usize("warmup", args.next())?,
        repetitions: parse_usize("repetitions", args.next())?,
        seed: parse_u64("seed", args.next())?,
    };
    if args.next().is_some() {
        return Err("too many arguments".to_owned());
    }
    if config.pages == 0 || config.signature_bits == 0 || config.workers == 0 {
        return Err("pages, signature_bits, and workers must be non-zero".to_owned());
    }
    if config.max_distance > config.signature_bits {
        return Err("max_distance must not exceed signature_bits".to_owned());
    }
    if config.repetitions == 0 {
        return Err("repetitions must be non-zero".to_owned());
    }
    Ok(config)
}

fn next_u64(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}

fn push_signature(state: &mut u64, signature_bits: usize, output: &mut Vec<u64>) {
    let words = signature_bits.div_ceil(64);
    for _ in 0..words {
        output.push(next_u64(state));
    }
    let tail_bits = signature_bits % 64;
    if tail_bits != 0 {
        let last = output.len() - 1;
        output[last] &= (1_u64 << tail_bits) - 1;
    }
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

fn scan_flat_parallel(
    signature_bits: usize,
    query: &[u64],
    flat_pages: &[u64],
    pages: usize,
    max_distance: usize,
    workers: usize,
) -> Result<Vec<usize>, String> {
    let words = signature_bits.div_ceil(64);
    if query.len() != words {
        return Err("query word count mismatch".to_owned());
    }
    if flat_pages.len() != pages.saturating_mul(words) {
        return Err("flat page buffer length mismatch".to_owned());
    }
    if workers == 0 {
        return Err("workers must be non-zero".to_owned());
    }

    let worker_count = workers.min(pages);
    let shard_pages = pages.div_ceil(worker_count);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for shard_start_page in (0..pages).step_by(shard_pages) {
            let shard_end_page = (shard_start_page + shard_pages).min(pages);
            handles.push(scope.spawn(move || {
                let mut selected = Vec::new();
                for page_id in shard_start_page..shard_end_page {
                    let start = page_id * words;
                    let end = start + words;
                    let page = &flat_pages[start..end];
                    let distance: u32 = query
                        .iter()
                        .zip(page.iter())
                        .map(|(a, b)| (a ^ b).count_ones())
                        .sum();
                    if distance as usize <= max_distance {
                        selected.push(page_id);
                    }
                }
                selected
            }));
        }

        let mut merged = Vec::new();
        for handle in handles {
            let mut local = handle
                .join()
                .map_err(|_| "worker thread panicked".to_owned())?;
            merged.append(&mut local);
        }
        Ok::<Vec<usize>, String>(merged)
    })
}

fn run(config: Config) -> Result<(), String> {
    let words = config.signature_bits.div_ceil(64);
    let mut state = config.seed;
    let mut query = Vec::with_capacity(words);
    push_signature(&mut state, config.signature_bits, &mut query);

    let mut flat_pages = Vec::with_capacity(config.pages.saturating_mul(words));
    for _ in 0..config.pages {
        push_signature(&mut state, config.signature_bits, &mut flat_pages);
    }

    let oracle_pages = flat_pages
        .chunks_exact(words)
        .map(|page| page.to_vec())
        .collect::<Vec<_>>();
    let scalar = scan_packed_pages(
        config.signature_bits,
        &query,
        &oracle_pages,
        config.max_distance,
    )
    .map_err(|error| format!("scalar oracle failed: {error:?}"))?;

    let flat = scan_flat_parallel(
        config.signature_bits,
        &query,
        &flat_pages,
        config.pages,
        config.max_distance,
        config.workers,
    )?;
    if flat != scalar.selected_pages {
        return Err("flat candidate set differs from scalar oracle".to_owned());
    }

    for _ in 0..config.warmup {
        black_box(scan_flat_parallel(
            config.signature_bits,
            &query,
            &flat_pages,
            config.pages,
            config.max_distance,
            config.workers,
        )?);
    }

    let mut samples = Vec::with_capacity(config.repetitions);
    for _ in 0..config.repetitions {
        let start = Instant::now();
        let result = scan_flat_parallel(
            config.signature_bits,
            &query,
            &flat_pages,
            config.pages,
            config.max_distance,
            config.workers,
        )?;
        black_box(result);
        samples.push(start.elapsed().as_nanos());
    }

    let min_ns = *samples.iter().min().expect("non-empty repetitions");
    let max_ns = *samples.iter().max().expect("non-empty repetitions");
    let median = median_ns(&mut samples);
    println!(
        "schema,seed,pages,signature_bits,max_distance,workers,warmup,repetitions,selected_pages,bits_compared,median_ns,min_ns,max_ns"
    );
    println!(
        "bkv-k3-flat-v1,{},{},{},{},{},{},{},{},{},{},{},{}",
        config.seed,
        config.pages,
        config.signature_bits,
        config.max_distance,
        config.workers,
        config.warmup,
        config.repetitions,
        scalar.selected_pages.len(),
        scalar.bits_compared,
        median,
        min_ns,
        max_ns
    );
    Ok(())
}

fn main() -> ExitCode {
    match parse_config().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!(
                "usage: bkv_flat_scan_bench <pages> <signature_bits> <max_distance> <workers> <warmup> <repetitions> <seed>"
            );
            ExitCode::FAILURE
        }
    }
}
