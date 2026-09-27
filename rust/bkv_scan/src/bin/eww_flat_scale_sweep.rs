#![forbid(unsafe_code)]

//! EWW-K4 host scaling sweep for FLAT's production W64 paged-KV map.
//!
//! The sweep measures only host-side metadata traversal and address lookup.
//! It does not execute numerical attention or establish GPU, DRAM/HBM, TTFT,
//! TPOT, tokens/s, or model-quality results.

use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

const SCHEMA: &str = "kvlab.eww-k4-flat-scale-sweep/v1";
const PAGE_COUNTS: [usize; 7] = [1, 4, 16, 64, 256, 1024, 4096];
const PAGE_SIZE_VARIANTS: [usize; 3] = [4, 16, 64];
const FIXED_PAGE_SIZE: usize = 16;
const FIXED_PAGE_COUNT: usize = 256;
const WARMUP: usize = 1;
const REPETITIONS: usize = 5;

fn build_table(page_size: usize, mapped_pages: usize) -> Result<PagedKvTable, String> {
    let live_tokens = page_size
        .checked_mul(mapped_pages)
        .ok_or_else(|| "live token count overflow".to_owned())?;
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size,
        physical_pages: mapped_pages,
    })
    .map_err(|error| error.to_string())?;
    table
        .append(live_tokens)
        .map_err(|error| error.to_string())?;
    Ok(table)
}

fn lookup_checksum(table: &PagedKvTable) -> Result<u64, String> {
    let mut checksum = 0_u64;
    for token in 0..table.len() {
        let address = table
            .address(token)
            .ok_or_else(|| format!("missing address for token {token}"))?;
        let page = u64::try_from(address.physical_page)
            .map_err(|_| "physical page does not fit u64".to_owned())?;
        let offset = u64::try_from(address.offset_in_page)
            .map_err(|_| "page offset does not fit u64".to_owned())?;
        checksum = checksum.rotate_left(7) ^ page.rotate_left(3) ^ offset ^ address.generation;
    }
    Ok(checksum)
}

fn lane_checksum(table: &PagedKvTable) -> u64 {
    table
        .mapped_page_lanes()
        .iter()
        .copied()
        .fold(table.generation(), |acc, lane| acc.rotate_left(5) ^ lane)
}

fn median(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2
    } else {
        samples[middle]
    }
}

fn time_lookup(table: &PagedKvTable) -> Result<u128, String> {
    for _ in 0..WARMUP {
        black_box(lookup_checksum(table)?);
    }
    let mut samples = Vec::with_capacity(REPETITIONS);
    for _ in 0..REPETITIONS {
        let start = Instant::now();
        black_box(lookup_checksum(table)?);
        samples.push(start.elapsed().as_nanos());
    }
    Ok(median(&mut samples))
}

fn time_lane_walk(table: &PagedKvTable) -> u128 {
    for _ in 0..WARMUP {
        black_box(lane_checksum(table));
    }
    let mut samples = Vec::with_capacity(REPETITIONS);
    for _ in 0..REPETITIONS {
        let start = Instant::now();
        black_box(lane_checksum(table));
        samples.push(start.elapsed().as_nanos());
    }
    median(&mut samples)
}

fn emit_case(axis: &str, page_size: usize, mapped_pages: usize) -> Result<(), String> {
    let table = build_table(page_size, mapped_pages)?;
    let payload_bytes = table
        .mapped_page_lanes()
        .len()
        .checked_mul(core::mem::size_of::<u64>())
        .and_then(|bytes| bytes.checked_add(core::mem::size_of::<u64>()))
        .ok_or_else(|| "payload byte accounting overflow".to_owned())?;
    let lookup_ns = time_lookup(&table)?;
    let lane_walk_ns = time_lane_walk(&table);
    let live_tokens = table.len();

    println!(
        "{SCHEMA},{axis},{page_size},{mapped_pages},{live_tokens},{payload_bytes},{lookup_ns},{lane_walk_ns},{WARMUP},{REPETITIONS}"
    );
    Ok(())
}

fn run() -> Result<(), String> {
    println!(
        "schema,axis,page_size,mapped_pages,live_tokens,declared_control_payload_bytes,lookup_median_ns,lane_walk_median_ns,warmup,repetitions"
    );

    for mapped_pages in PAGE_COUNTS {
        emit_case("page-count", FIXED_PAGE_SIZE, mapped_pages)?;
    }
    for page_size in PAGE_SIZE_VARIANTS {
        emit_case("page-size", page_size, FIXED_PAGE_COUNT)?;
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_matrix_is_strictly_positive_and_ordered() {
        assert!(PAGE_COUNTS.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(PAGE_SIZE_VARIANTS.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(PAGE_COUNTS.iter().all(|value| *value > 0));
        assert!(PAGE_SIZE_VARIANTS.iter().all(|value| *value > 0));
    }

    #[test]
    fn production_payload_is_one_w64_lane_per_page_plus_one_epoch() {
        let table = build_table(16, 64).unwrap();
        assert_eq!(table.mapped_page_lanes().len(), 64);
        assert_eq!(
            core::mem::size_of_val(table.mapped_page_lanes()) + core::mem::size_of::<u64>(),
            64 * 8 + 8
        );
    }

    #[test]
    fn lookup_and_lane_checksums_are_deterministic() {
        let table = build_table(4, 8).unwrap();
        let lookup = lookup_checksum(&table).unwrap();
        let lanes = lane_checksum(&table);
        assert_eq!(lookup, lookup_checksum(&table).unwrap());
        assert_eq!(lanes, lane_checksum(&table));
    }
}
