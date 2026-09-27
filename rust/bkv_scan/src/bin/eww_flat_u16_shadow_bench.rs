#![forbid(unsafe_code)]

//! EWW-K4c host comparison of FLAT W64 -> W32 production projection and packed-u16 shadow.
//!
//! This benchmark does not execute WGSL. The packed-u16 path remains a host-only
//! shadow until the shader/uniform contract is migrated and separately qualified.

use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::{
    WgpuPackedPagedKvTable16, WgpuPagedKvTable, WGSL_PAGED_U16_MAX_PHYSICAL_PAGES,
};
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

const SCHEMA: &str = "kvlab.eww-k4c-packed-u16-shadow/v1";
const FLAT_REVISION: &str = "512224212ad834f19c407878ff4e6569fdcbe508";
const PAGE_COUNTS: [usize; 5] = [1, 4, 16, 64, 256];
const PAGE_SIZE: usize = 16;
const WARMUP: usize = 2;
const REPETITIONS: usize = 9;

fn build_table(mapped_pages: usize) -> Result<PagedKvTable, String> {
    let live_tokens = mapped_pages
        .checked_mul(PAGE_SIZE)
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

fn measure<T>(
    warmup: usize,
    repetitions: usize,
    mut f: impl FnMut() -> Result<T, String>,
) -> Result<(u128, u128, u128, T), String> {
    for _ in 0..warmup {
        black_box(f()?);
    }
    let mut samples = Vec::with_capacity(repetitions);
    let mut last = None;
    for _ in 0..repetitions {
        let start = Instant::now();
        let value = f()?;
        samples.push(start.elapsed().as_nanos());
        last = Some(value);
    }
    let min = *samples.iter().min().expect("non-empty repetitions");
    let max = *samples.iter().max().expect("non-empty repetitions");
    let med = median(&mut samples);
    Ok((med, min, max, last.expect("non-empty repetitions")))
}

fn run() -> Result<(), String> {
    if PAGE_COUNTS
        .iter()
        .any(|pages| *pages > WGSL_PAGED_U16_MAX_PHYSICAL_PAGES)
    {
        return Err("frozen K4c sweep exceeds packed-u16 physical-page domain".to_owned());
    }

    println!(
        "schema,flat_revision,mapped_pages,host_w64_bytes,w32_page_map_bytes,u16_packed_bytes,u16_theoretical_uniform_bytes,w32_projection_median_ns,u16_projection_median_ns,w32_projection_min_ns,u16_projection_min_ns,w32_projection_max_ns,u16_projection_max_ns,gpu_status,u16_shader_status"
    );

    for mapped_pages in PAGE_COUNTS {
        let table = build_table(mapped_pages)?;

        let (w32_med, w32_min, w32_max, w32) = measure(WARMUP, REPETITIONS, || {
            WgpuPagedKvTable::from_table(&table).map_err(|error| error.to_string())
        })?;
        let (u16_med, u16_min, u16_max, packed) = measure(WARMUP, REPETITIONS, || {
            WgpuPackedPagedKvTable16::from_table(&table).map_err(|error| error.to_string())
        })?;

        if w32.entries().len() != mapped_pages || packed.mapped_pages() != mapped_pages {
            return Err("projection page-count drift".to_owned());
        }
        for logical_page in 0..mapped_pages {
            let expected =
                u16::try_from(logical_page).map_err(|_| "logical page does not fit u16")?;
            if packed.physical_page(logical_page) != Some(expected) {
                return Err(format!(
                    "packed-u16 identity drift at logical page {logical_page}"
                ));
            }
            if w32.entries()[logical_page] != u32::from(expected) {
                return Err(format!("W32 identity drift at logical page {logical_page}"));
            }
        }

        let host_w64_bytes = core::mem::size_of_val(table.mapped_page_lanes());
        let w32_page_map_bytes = core::mem::size_of_val(w32.entries());
        let u16_packed_bytes = packed.packed_page_map_bytes();
        let u16_theoretical_uniform_bytes = packed.theoretical_encoded_uniform_bytes();

        println!(
            "{SCHEMA},{FLAT_REVISION},{mapped_pages},{host_w64_bytes},{w32_page_map_bytes},{u16_packed_bytes},{u16_theoretical_uniform_bytes},{w32_med},{u16_med},{w32_min},{u16_min},{w32_max},{u16_max},not-executed,shadow-only"
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
    fn packed_u16_payload_is_half_w32_for_even_page_counts() {
        for mapped_pages in [4, 16, 64, 256] {
            let table = build_table(mapped_pages).unwrap();
            let w32 = WgpuPagedKvTable::from_table(&table).unwrap();
            let packed = WgpuPackedPagedKvTable16::from_table(&table).unwrap();

            assert_eq!(
                packed.packed_page_map_bytes() * 2,
                core::mem::size_of_val(w32.entries())
            );
        }
    }

    #[test]
    fn odd_page_count_uses_one_padded_halfword_only() {
        let table = build_table(1).unwrap();
        let packed = WgpuPackedPagedKvTable16::from_table(&table).unwrap();
        assert_eq!(packed.packed_entries().len(), 1);
        assert_eq!(packed.packed_page_map_bytes(), 4);
        assert_eq!(packed.physical_page(0), Some(0));
    }

    #[test]
    fn theoretical_fixed_uniform_is_560_bytes() {
        assert_eq!(flat_attention::WGSL_PAGED_U16_UNIFORM_U32 * 4, 560);
    }
}
