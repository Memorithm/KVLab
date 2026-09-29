#![forbid(unsafe_code)]

//! EWW-K4g host oracle for the packed-u16 WGSL page-table addressing contract.
//!
//! The oracle consumes FLAT's actual packed-u16 shadow uniform words and mirrors
//! only the shader's physical_page(logical_page) halfword extraction. It does
//! not create a GPU device, compile WGSL, or measure performance.

use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::{WgpuPackedPagedKvTable16, WGSL_PAGED_U16_UNIFORM_U32};
use std::process::ExitCode;

const SCHEMA: &str = "kvlab.eww-k4g-u16-shader-contract/v1";
const FLAT_REVISION: &str = "dff65361cb39b91e88a6d2bc99fc5469187809f4";
const HEADER_WORDS: usize = 12;
const MAX_LOGICAL_PAGES: usize = 256;

fn build_table(mapped_pages: usize) -> Result<PagedKvTable, String> {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 1,
        physical_pages: mapped_pages,
    })
    .map_err(|error| error.to_string())?;
    table
        .append(mapped_pages)
        .map_err(|error| error.to_string())?;
    Ok(table)
}

fn shader_shadow_physical_page(words: &[u32], logical_page: usize) -> Result<u16, String> {
    if logical_page >= MAX_LOGICAL_PAGES {
        return Err(format!(
            "logical page {logical_page} exceeds shadow shader limit {MAX_LOGICAL_PAGES}"
        ));
    }
    if words.len() != WGSL_PAGED_U16_UNIFORM_U32 {
        return Err(format!(
            "shadow uniform length {} != {}",
            words.len(),
            WGSL_PAGED_U16_UNIFORM_U32
        ));
    }

    let packed_word_index = logical_page / 2;
    let word = words[HEADER_WORDS + packed_word_index];
    let shift = (logical_page % 2) * 16;
    Ok(((word >> shift) & u32::from(u16::MAX)) as u16)
}

fn verify_case(mapped_pages: usize) -> Result<(), String> {
    let table = build_table(mapped_pages)?;
    let packed =
        WgpuPackedPagedKvTable16::from_table(&table).map_err(|error| error.to_string())?;
    let words = packed
        .shadow_uniform_words([0; 8])
        .map_err(|error| error.to_string())?;

    if words.len() != WGSL_PAGED_U16_UNIFORM_U32 {
        return Err("unexpected shadow uniform length".to_owned());
    }
    if usize::try_from(words[3]).map_err(|_| "mapped_pages header does not fit usize")?
        != mapped_pages
    {
        return Err(format!("mapped-page header drift for case {mapped_pages}"));
    }

    for logical_page in 0..mapped_pages {
        let expected = packed
            .physical_page(logical_page)
            .ok_or_else(|| format!("missing packed mapping {logical_page}"))?;
        let decoded = shader_shadow_physical_page(&words, logical_page)?;
        if decoded != expected {
            return Err(format!(
                "shader halfword parity drift pages={mapped_pages} logical={logical_page}: decoded={decoded} expected={expected}"
            ));
        }
    }

    // For odd page counts, the unused high halfword of the final packed word
    // must remain zero so it cannot accidentally encode a phantom page.
    if mapped_pages % 2 == 1 {
        let last_word = words[HEADER_WORDS + mapped_pages / 2];
        if (last_word >> 16) != 0 {
            return Err(format!(
                "odd-page high-half padding is non-zero for mapped_pages={mapped_pages}"
            ));
        }
    }

    Ok(())
}

fn run() -> Result<(), String> {
    println!("schema,flat_revision,cases,max_logical_pages,uniform_words,status");
    for mapped_pages in 1..=MAX_LOGICAL_PAGES {
        verify_case(mapped_pages)?;
    }
    println!(
        "{SCHEMA},{FLAT_REVISION},{MAX_LOGICAL_PAGES},{MAX_LOGICAL_PAGES},{WGSL_PAGED_U16_UNIFORM_U32},exact-host-parity"
    );
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
    fn every_portable_page_count_replays_shader_halfword_contract() {
        for mapped_pages in 1..=MAX_LOGICAL_PAGES {
            verify_case(mapped_pages).unwrap();
        }
    }

    #[test]
    fn low_and_high_halfwords_decode_distinct_neighbor_pages() {
        let table = build_table(2).unwrap();
        let packed = WgpuPackedPagedKvTable16::from_table(&table).unwrap();
        let words = packed.shadow_uniform_words([0; 8]).unwrap();

        assert_eq!(shader_shadow_physical_page(&words, 0).unwrap(), 0);
        assert_eq!(shader_shadow_physical_page(&words, 1).unwrap(), 1);
        assert_eq!(words[HEADER_WORDS], 1 << 16);
    }

    #[test]
    fn malformed_uniform_length_fails_closed() {
        let error = shader_shadow_physical_page(&[0; 16], 0).unwrap_err();
        assert!(error.contains("shadow uniform length"));
    }

    #[test]
    fn logical_page_above_portable_limit_fails_closed() {
        let words = vec![0; WGSL_PAGED_U16_UNIFORM_U32];
        let error = shader_shadow_physical_page(&words, MAX_LOGICAL_PAGES).unwrap_err();
        assert!(error.contains("exceeds shadow shader limit"));
    }
}
