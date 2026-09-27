#![forbid(unsafe_code)]

//! EWW-K4b host construction cost for FLAT's production W64 -> W32 WGPU map.
//!
//! This benchmark constructs the actual WgpuPagedKvTable host descriptor but
//! does not create a GPU device or execute WGSL.

use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::WgpuPagedKvTable;

const WGSL_PAGED_MAX_LOGICAL_PAGES_PINNED_PINNED: usize = 256;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

const SCHEMA: &str = "kvlab.eww-k4b-flat-w64-to-w32/v1";
const FLAT_REVISION: &str = "7bd9f1a2254d10329861e41a11694e7b4cffdd6b";
const PAGE_COUNTS: [usize; 5] = [1, 4, 16, 64, 256];
const PAGE_SIZE: usize = 16;
const WARMUP: usize = 2;
const REPETITIONS: usize = 9;

fn build_table(mapped_pages: usize) -> Result<PagedKvTable, String> {
    let live_tokens = PAGE_SIZE
        .checked_mul(mapped_pages)
        .ok_or_else(|| "live token count overflow".to_owned())?;
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: PAGE_SIZE,
        physical_pages: mapped_pages,
    })
    .map_err(|error| error.to_string())?;
    table
        .append(live_tokens)
        .map_err(|error| error.to_string())?;
    Ok(table)
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

fn time_projection(table: &PagedKvTable) -> Result<(u128, u128, u128, WgpuPagedKvTable), String> {
    for _ in 0..WARMUP {
        black_box(WgpuPagedKvTable::from_table(table).map_err(|error| error.to_string())?);
    }

    let mut samples = Vec::with_capacity(REPETITIONS);
    let mut last = None;
    for _ in 0..REPETITIONS {
        let start = Instant::now();
        let projected = WgpuPagedKvTable::from_table(table).map_err(|error| error.to_string())?;
        let elapsed = start.elapsed().as_nanos();
        black_box(projected.entries());
        last = Some(projected);
        samples.push(elapsed);
    }

    let minimum = *samples.iter().min().expect("non-empty repetitions");
    let maximum = *samples.iter().max().expect("non-empty repetitions");
    let med = median(&mut samples);
    Ok((med, minimum, maximum, last.expect("non-empty repetitions")))
}

fn run() -> Result<(), String> {
    println!(
        "schema,flat_revision,page_size,mapped_pages,host_w64_page_map_bytes,device_w32_page_map_bytes,projection_median_ns,projection_min_ns,projection_max_ns,generation,uniform_encoded_bytes_status,gpu_status"
    );

    for mapped_pages in PAGE_COUNTS {
        let table = build_table(mapped_pages)?;
        let (median_ns, min_ns, max_ns, device) = time_projection(&table)?;
        let host_w64_page_map_bytes = core::mem::size_of_val(table.mapped_page_lanes());
        let device_w32_page_map_bytes = core::mem::size_of_val(device.entries());

        if device.entries().len() != mapped_pages {
            return Err(format!(
                "W32 projection page count drift: {} != {mapped_pages}",
                device.entries().len()
            ));
        }
        for (expected, observed) in device.entries().iter().copied().enumerate() {
            let expected =
                u32::try_from(expected).map_err(|_| "expected page index does not fit u32")?;
            if observed != expected {
                return Err(format!(
                    "W32 projection identity drift: observed {observed}, expected {expected}"
                ));
            }
        }

        println!(
            "{SCHEMA},{FLAT_REVISION},{PAGE_SIZE},{mapped_pages},{host_w64_page_map_bytes},{device_w32_page_map_bytes},{median_ns},{min_ns},{max_ns},{},not-exposed-by-pinned-revision,not-executed",
            device.generation()
        );
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
    fn frozen_page_counts_fit_portable_uniform_limit() {
        assert!(PAGE_COUNTS
            .iter()
            .all(|mapped_pages| *mapped_pages <= WGSL_PAGED_MAX_LOGICAL_PAGES_PINNED));
        assert_eq!(*PAGE_COUNTS.last().unwrap(), WGSL_PAGED_MAX_LOGICAL_PAGES_PINNED);
    }

    #[test]
    fn actual_projection_halves_page_map_element_payload_width() {
        let table = build_table(16).unwrap();
        let device = WgpuPagedKvTable::from_table(&table).unwrap();
        assert_eq!(table.mapped_page_lanes().len(), 16);
        assert_eq!(device.entries().len(), 16);
        assert_eq!(
            core::mem::size_of_val(table.mapped_page_lanes()),
            16 * core::mem::size_of::<u64>()
        );
        assert_eq!(
            core::mem::size_of_val(device.entries()),
            16 * core::mem::size_of::<u32>()
        );
    }

    #[test]
    fn epoch_generation_remains_outside_per_page_w32_entries() {
        let mut table = build_table(4).unwrap();
        let before = WgpuPagedKvTable::from_table(&table).unwrap();
        table.reset().unwrap();
        table.append(PAGE_SIZE * 4).unwrap();
        let after = WgpuPagedKvTable::from_table(&table).unwrap();

        assert_eq!(before.entries(), after.entries());
        assert_eq!(before.generation(), 0);
        assert_eq!(after.generation(), 1);
    }
}
