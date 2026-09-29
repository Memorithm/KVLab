#![forbid(unsafe_code)]

//! EWW-K4f host materialization cost for FLAT's packed-u16 shadow uniform.
//!
//! This benchmark materializes the exact 140-u32 / 560-byte host shadow
//! uniform provided by FLAT #322. The feature first landed at
//! `5be5c0084db51e86c91a899bfeecf111ee3f52e7`; this binary links the exact
//! dependency revision recorded separately below. It does not submit a GPU
//! command or execute the production W32 shader.

use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::{WgpuPackedPagedKvTable16, WGSL_PAGED_U16_UNIFORM_U32};
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

const SCHEMA: &str = "kvlab.eww-k4f-u16-uniform-materialization/v2";
const FLAT_FEATURE_REVISION: &str = "5be5c0084db51e86c91a899bfeecf111ee3f52e7";
const FLAT_LINKED_REVISION: &str = "dff65361cb39b91e88a6d2bc99fc5469187809f4";
const PAGE_COUNTS: [usize; 5] = [1, 4, 16, 64, 256];
const PAGE_SIZE: usize = 16;
const HEADER_TAIL: [u32; 8] = [64, 8, 2, 0x3f00_0000, 0x447a_0000, 16, 0, 0];
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

fn time_uniform(packed: &WgpuPackedPagedKvTable16) -> Result<(u128, u128, u128, Vec<u32>), String> {
    for _ in 0..WARMUP {
        black_box(
            packed
                .shadow_uniform_words(HEADER_TAIL)
                .map_err(|error| error.to_string())?,
        );
    }

    let mut samples = Vec::with_capacity(REPETITIONS);
    let mut last = None;
    for _ in 0..REPETITIONS {
        let start = Instant::now();
        let words = packed
            .shadow_uniform_words(HEADER_TAIL)
            .map_err(|error| error.to_string())?;
        samples.push(start.elapsed().as_nanos());
        black_box(words.as_slice());
        last = Some(words);
    }

    let min = *samples.iter().min().expect("non-empty repetitions");
    let max = *samples.iter().max().expect("non-empty repetitions");
    let med = median(&mut samples);
    Ok((med, min, max, last.expect("non-empty repetitions")))
}

fn run() -> Result<(), String> {
    println!(
        "schema,flat_feature_revision,flat_linked_revision,mapped_pages,packed_page_map_bytes,uniform_bytes,used_page_words,padded_page_words,materialize_median_ns,materialize_min_ns,materialize_max_ns,gpu_status,shader_status"
    );

    for mapped_pages in PAGE_COUNTS {
        let table = build_table(mapped_pages)?;
        let packed =
            WgpuPackedPagedKvTable16::from_table(&table).map_err(|error| error.to_string())?;
        let (median_ns, min_ns, max_ns, words) = time_uniform(&packed)?;

        if words.len() != WGSL_PAGED_U16_UNIFORM_U32 {
            return Err(format!(
                "uniform word count drift: {} != {}",
                words.len(),
                WGSL_PAGED_U16_UNIFORM_U32
            ));
        }
        if words[..4]
            != [
                u32::try_from(table.len()).map_err(|_| "live tokens do not fit u32")?,
                u32::try_from(PAGE_SIZE).map_err(|_| "page size does not fit u32")?,
                u32::try_from(mapped_pages).map_err(|_| "physical pages do not fit u32")?,
                u32::try_from(mapped_pages).map_err(|_| "mapped pages do not fit u32")?,
            ]
        {
            return Err("shadow uniform table header drift".to_owned());
        }
        if &words[4..12] != HEADER_TAIL.as_slice() {
            return Err("shadow uniform decode header tail drift".to_owned());
        }
        let used_page_words = packed.packed_entries().len();
        if &words[12..12 + used_page_words] != packed.packed_entries() {
            return Err("shadow uniform packed page-map drift".to_owned());
        }
        if words[12 + used_page_words..].iter().any(|word| *word != 0) {
            return Err("shadow uniform padding is non-zero".to_owned());
        }

        let uniform_bytes = core::mem::size_of_val(words.as_slice());
        if uniform_bytes != packed.theoretical_encoded_uniform_bytes() {
            return Err("materialized/theoretical uniform byte count drift".to_owned());
        }
        let padded_page_words = WGSL_PAGED_U16_UNIFORM_U32
            .checked_sub(12 + used_page_words)
            .ok_or_else(|| "uniform padding accounting underflow".to_owned())?;

        println!(
            "{SCHEMA},{FLAT_FEATURE_REVISION},{FLAT_LINKED_REVISION},{mapped_pages},{},{uniform_bytes},{used_page_words},{padded_page_words},{median_ns},{min_ns},{max_ns},not-executed,shadow-only",
            packed.packed_page_map_bytes(),
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
    fn full_shadow_uniform_is_exactly_560_bytes() {
        let table = build_table(16).unwrap();
        let packed = WgpuPackedPagedKvTable16::from_table(&table).unwrap();
        let words = packed.shadow_uniform_words(HEADER_TAIL).unwrap();

        assert_eq!(words.len(), WGSL_PAGED_U16_UNIFORM_U32);
        assert_eq!(core::mem::size_of_val(words.as_slice()), 560);
    }

    #[test]
    fn page_map_payload_scales_while_uniform_envelope_stays_fixed() {
        for mapped_pages in PAGE_COUNTS {
            let table = build_table(mapped_pages).unwrap();
            let packed = WgpuPackedPagedKvTable16::from_table(&table).unwrap();
            let words = packed.shadow_uniform_words(HEADER_TAIL).unwrap();

            assert_eq!(core::mem::size_of_val(words.as_slice()), 560);
            assert_eq!(
                packed.packed_page_map_bytes(),
                mapped_pages.div_ceil(2) * core::mem::size_of::<u32>()
            );
        }
    }

    #[test]
    fn odd_page_count_retains_one_zero_padded_halfword() {
        let table = build_table(1).unwrap();
        let packed = WgpuPackedPagedKvTable16::from_table(&table).unwrap();
        assert_eq!(packed.packed_entries(), &[0]);
        let words = packed.shadow_uniform_words(HEADER_TAIL).unwrap();
        assert_eq!(words[12], 0);
    }
}
