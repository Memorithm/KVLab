#![forbid(unsafe_code)]

//! EWW-K4c structural utilization of FLAT's fixed portable WGPU paged uniform.
//!
//! This harness distinguishes useful W32 page-map payload from the exact fixed
//! encoded uniform bytes exposed by FLAT. It does not allocate a GPU resource,
//! submit work, or claim physical device-memory savings.

use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::WgpuPagedKvTable;
use kvlab_bkv_scan::provenance::FLAT_ATTENTION_EXECUTED_REVISION;
use std::process::ExitCode;

const SCHEMA: &str = "kvlab.eww-k4c-w32-uniform-utilization/v2";
const CONTRACT_ORIGIN_REVISION: &str = "5c49c1a88198c58390826f8cc1a6539ea52874a4";
const PAGE_COUNTS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const PAGE_SIZE: usize = 16;

fn build_table(mapped_pages: usize) -> Result<PagedKvTable, String> {
    let live_tokens = PAGE_SIZE
        .checked_mul(mapped_pages)
        .ok_or_else(|| "live-token count overflow".to_owned())?;
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

fn utilization_basis_points(useful_bytes: usize, encoded_bytes: usize) -> Result<u64, String> {
    if encoded_bytes == 0 {
        return Err("encoded uniform byte count must be non-zero".to_owned());
    }
    let useful = u64::try_from(useful_bytes).map_err(|_| "useful byte count does not fit u64")?;
    let encoded =
        u64::try_from(encoded_bytes).map_err(|_| "encoded byte count does not fit u64")?;
    useful
        .checked_mul(10_000)
        .map(|scaled| scaled / encoded)
        .ok_or_else(|| "uniform utilization calculation overflow".to_owned())
}

fn run() -> Result<(), String> {
    println!(
        "schema,contract_origin_revision,executed_dependency_revision,page_size,mapped_pages,host_w64_page_map_bytes,useful_device_w32_page_map_bytes,fixed_encoded_uniform_bytes,unused_encoded_uniform_bytes,useful_page_map_basis_points,generation,gpu_status"
    );

    let mut expected_uniform_bytes = None;
    for mapped_pages in PAGE_COUNTS {
        let table = build_table(mapped_pages)?;
        let projected = WgpuPagedKvTable::from_table(&table).map_err(|error| error.to_string())?;

        let host_w64_page_map_bytes = core::mem::size_of_val(table.mapped_page_lanes());
        let useful_device_w32_page_map_bytes = projected.page_map_payload_bytes();
        let fixed_encoded_uniform_bytes = projected.encoded_uniform_bytes();

        match expected_uniform_bytes {
            None => expected_uniform_bytes = Some(fixed_encoded_uniform_bytes),
            Some(expected) if expected == fixed_encoded_uniform_bytes => {}
            Some(expected) => {
                return Err(format!(
                    "fixed uniform size drifted from {expected} to {fixed_encoded_uniform_bytes}"
                ));
            }
        }

        if useful_device_w32_page_map_bytes > fixed_encoded_uniform_bytes {
            return Err(format!(
                "useful W32 page map {useful_device_w32_page_map_bytes} exceeds fixed uniform {fixed_encoded_uniform_bytes}"
            ));
        }
        let unused_encoded_uniform_bytes =
            fixed_encoded_uniform_bytes - useful_device_w32_page_map_bytes;
        let useful_page_map_basis_points = utilization_basis_points(
            useful_device_w32_page_map_bytes,
            fixed_encoded_uniform_bytes,
        )?;

        println!(
            "{SCHEMA},{CONTRACT_ORIGIN_REVISION},{FLAT_ATTENTION_EXECUTED_REVISION},{PAGE_SIZE},{mapped_pages},{host_w64_page_map_bytes},{useful_device_w32_page_map_bytes},{fixed_encoded_uniform_bytes},{unused_encoded_uniform_bytes},{useful_page_map_basis_points},{},not-executed",
            projected.generation()
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
    fn fixed_uniform_stays_constant_while_useful_payload_scales() {
        let one = WgpuPagedKvTable::from_table(&build_table(1).unwrap()).unwrap();
        let sixty_four = WgpuPagedKvTable::from_table(&build_table(64).unwrap()).unwrap();
        let full = WgpuPagedKvTable::from_table(&build_table(256).unwrap()).unwrap();

        assert_eq!(one.encoded_uniform_bytes(), 1072);
        assert_eq!(sixty_four.encoded_uniform_bytes(), 1072);
        assert_eq!(full.encoded_uniform_bytes(), 1072);

        assert_eq!(one.page_map_payload_bytes(), 4);
        assert_eq!(sixty_four.page_map_payload_bytes(), 256);
        assert_eq!(full.page_map_payload_bytes(), 1024);
    }

    #[test]
    fn useful_page_map_utilization_is_monotonic_for_frozen_sweep() {
        let mut previous = 0_u64;
        for mapped_pages in PAGE_COUNTS {
            let projected =
                WgpuPagedKvTable::from_table(&build_table(mapped_pages).unwrap()).unwrap();
            let utilization = utilization_basis_points(
                projected.page_map_payload_bytes(),
                projected.encoded_uniform_bytes(),
            )
            .unwrap();
            assert!(utilization >= previous);
            previous = utilization;
        }
    }

    #[test]
    fn page_map_width_is_exactly_half_between_host_and_device_projection() {
        for mapped_pages in [1_usize, 16, 64, 256] {
            let table = build_table(mapped_pages).unwrap();
            let projected = WgpuPagedKvTable::from_table(&table).unwrap();
            assert_eq!(
                core::mem::size_of_val(table.mapped_page_lanes()),
                projected.page_map_payload_bytes() * 2
            );
        }
    }
}
