#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Debug, Clone)]
struct Config {
    total_pages: usize,
    signature_bits: usize,
    max_distance: usize,
    page_offset: usize,
    shard_pages: usize,
    workers: usize,
    warmup: usize,
    repetitions: usize,
    seed: u64,
    selected_out: PathBuf,
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
        total_pages: parse_usize("total_pages", args.next())?,
        signature_bits: parse_usize("signature_bits", args.next())?,
        max_distance: parse_usize("max_distance", args.next())?,
        page_offset: parse_usize("page_offset", args.next())?,
        shard_pages: parse_usize("shard_pages", args.next())?,
        workers: parse_usize("workers", args.next())?,
        warmup: parse_usize("warmup", args.next())?,
        repetitions: parse_usize("repetitions", args.next())?,
        seed: parse_u64("seed", args.next())?,
        selected_out: PathBuf::from(
            args.next()
                .ok_or_else(|| "missing selected_out".to_owned())?,
        ),
    };
    if args.next().is_some() {
        return Err("too many arguments".to_owned());
    }
    validate_config(&config)?;
    Ok(config)
}

fn validate_config(config: &Config) -> Result<(), String> {
    if config.total_pages == 0
        || config.signature_bits == 0
        || config.shard_pages == 0
        || config.workers == 0
    {
        return Err(
            "total_pages, signature_bits, shard_pages, and workers must be non-zero".to_owned(),
        );
    }
    if config.max_distance > config.signature_bits {
        return Err("max_distance must not exceed signature_bits".to_owned());
    }
    if config.repetitions == 0 {
        return Err("repetitions must be non-zero".to_owned());
    }
    let end = config
        .page_offset
        .checked_add(config.shard_pages)
        .ok_or_else(|| "page range overflow".to_owned())?;
    if end > config.total_pages {
        return Err("page_offset + shard_pages exceeds total_pages".to_owned());
    }
    Ok(())
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

fn build_query_and_shard(
    total_pages: usize,
    signature_bits: usize,
    page_offset: usize,
    shard_pages: usize,
    seed: u64,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let end = page_offset
        .checked_add(shard_pages)
        .ok_or_else(|| "page range overflow".to_owned())?;
    if end > total_pages {
        return Err("page range exceeds total_pages".to_owned());
    }

    let words = signature_bits.div_ceil(64);
    let mut state = seed;
    let mut query = Vec::with_capacity(words);
    push_signature(&mut state, signature_bits, &mut query);

    let skip_words = page_offset
        .checked_mul(words)
        .ok_or_else(|| "page offset word count overflow".to_owned())?;
    for _ in 0..skip_words {
        black_box(next_u64(&mut state));
    }

    let total_words = shard_pages
        .checked_mul(words)
        .ok_or_else(|| "shard word count overflow".to_owned())?;
    let mut flat_pages = Vec::with_capacity(total_words);
    for _ in 0..shard_pages {
        push_signature(&mut state, signature_bits, &mut flat_pages);
    }
    Ok((query, flat_pages))
}

fn validate_layout(
    signature_bits: usize,
    query: &[u64],
    flat_pages: &[u64],
    shard_pages: usize,
) -> Result<usize, String> {
    let words = signature_bits.div_ceil(64);
    if query.len() != words {
        return Err("query word count mismatch".to_owned());
    }
    let expected = shard_pages
        .checked_mul(words)
        .ok_or_else(|| "flat buffer length overflow".to_owned())?;
    if flat_pages.len() != expected {
        return Err("flat buffer length mismatch".to_owned());
    }
    Ok(words)
}

fn scan_scalar(
    signature_bits: usize,
    query: &[u64],
    flat_pages: &[u64],
    page_offset: usize,
    shard_pages: usize,
    max_distance: usize,
) -> Result<Vec<usize>, String> {
    let words = validate_layout(signature_bits, query, flat_pages, shard_pages)?;
    let mut selected = Vec::new();
    for local_page in 0..shard_pages {
        let start = local_page * words;
        let page = &flat_pages[start..start + words];
        let distance: u32 = query
            .iter()
            .zip(page.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        if distance as usize <= max_distance {
            selected.push(page_offset + local_page);
        }
    }
    Ok(selected)
}

fn scan_parallel(
    signature_bits: usize,
    query: &[u64],
    flat_pages: &[u64],
    page_offset: usize,
    shard_pages: usize,
    max_distance: usize,
    workers: usize,
) -> Result<Vec<usize>, String> {
    let words = validate_layout(signature_bits, query, flat_pages, shard_pages)?;
    let worker_count = workers.min(shard_pages);
    let shard_span = shard_pages.div_ceil(worker_count);

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for local_start in (0..shard_pages).step_by(shard_span) {
            let local_end = (local_start + shard_span).min(shard_pages);
            handles.push(scope.spawn(move || {
                let mut selected = Vec::new();
                for local_page in local_start..local_end {
                    let start = local_page * words;
                    let page = &flat_pages[start..start + words];
                    let distance: u32 = query
                        .iter()
                        .zip(page.iter())
                        .map(|(a, b)| (a ^ b).count_ones())
                        .sum();
                    if distance as usize <= max_distance {
                        selected.push(page_offset + local_page);
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

fn median_ns(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2
    } else {
        samples[middle]
    }
}

fn write_selected(path: &PathBuf, selected: &[usize]) -> Result<(), String> {
    let mut text = String::new();
    for page_id in selected {
        text.push_str(&page_id.to_string());
        text.push('\n');
    }
    fs::write(path, text).map_err(|error| format!("{}: {error}", path.display()))
}

fn run(config: Config) -> Result<(), String> {
    let (query, flat_pages) = build_query_and_shard(
        config.total_pages,
        config.signature_bits,
        config.page_offset,
        config.shard_pages,
        config.seed,
    )?;

    let scalar = scan_scalar(
        config.signature_bits,
        &query,
        &flat_pages,
        config.page_offset,
        config.shard_pages,
        config.max_distance,
    )?;
    let parallel = scan_parallel(
        config.signature_bits,
        &query,
        &flat_pages,
        config.page_offset,
        config.shard_pages,
        config.max_distance,
        config.workers,
    )?;
    if parallel != scalar {
        return Err("parallel distinct-shard candidates differ from scalar oracle".to_owned());
    }
    write_selected(&config.selected_out, &scalar)?;

    for _ in 0..config.warmup {
        black_box(scan_parallel(
            config.signature_bits,
            &query,
            &flat_pages,
            config.page_offset,
            config.shard_pages,
            config.max_distance,
            config.workers,
        )?);
    }

    let mut samples = Vec::with_capacity(config.repetitions);
    for _ in 0..config.repetitions {
        let start = Instant::now();
        black_box(scan_parallel(
            config.signature_bits,
            &query,
            &flat_pages,
            config.page_offset,
            config.shard_pages,
            config.max_distance,
            config.workers,
        )?);
        samples.push(start.elapsed().as_nanos());
    }

    let min_ns = *samples.iter().min().expect("non-empty repetitions");
    let max_ns = *samples.iter().max().expect("non-empty repetitions");
    let median = median_ns(&mut samples);
    let bits_compared = config
        .shard_pages
        .checked_mul(config.signature_bits)
        .ok_or_else(|| "bits_compared overflow".to_owned())?;

    println!("schema,seed,total_pages,page_offset,shard_pages,signature_bits,max_distance,workers,warmup,repetitions,selected_pages,bits_compared,median_ns,min_ns,max_ns");
    println!(
        "bkv-k4-distinct-shard-v1,{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        config.seed,
        config.total_pages,
        config.page_offset,
        config.shard_pages,
        config.signature_bits,
        config.max_distance,
        config.workers,
        config.warmup,
        config.repetitions,
        scalar.len(),
        bits_compared,
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
            eprintln!("usage: bkv_distinct_shard_bench <total_pages> <signature_bits> <max_distance> <page_offset> <shard_pages> <workers> <warmup> <repetitions> <seed> <selected_out>");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selected_for(total: usize, offset: usize, pages: usize) -> Vec<usize> {
        let (query, flat) = build_query_and_shard(total, 65, offset, pages, 7).unwrap();
        scan_scalar(65, &query, &flat, offset, pages, 32).unwrap()
    }

    #[test]
    fn distinct_shards_union_matches_monolithic_stream() {
        let total = 37;
        let split = 19;
        let monolithic = selected_for(total, 0, total);
        let mut merged = selected_for(total, 0, split);
        merged.extend(selected_for(total, split, total - split));
        assert_eq!(merged, monolithic);
    }

    #[test]
    fn parallel_distinct_shard_matches_scalar() {
        let total = 41;
        let offset = 13;
        let pages = 17;
        let (query, flat) = build_query_and_shard(total, 128, offset, pages, 11).unwrap();
        let scalar = scan_scalar(128, &query, &flat, offset, pages, 64).unwrap();
        for workers in [1, 2, 4, 8, 32] {
            let parallel = scan_parallel(128, &query, &flat, offset, pages, 64, workers).unwrap();
            assert_eq!(parallel, scalar);
        }
    }

    #[test]
    fn rejects_page_range_beyond_total() {
        let error = build_query_and_shard(10, 64, 8, 3, 1).unwrap_err();
        assert!(error.contains("exceeds total_pages") || error.contains("exceeds"));
    }
}
