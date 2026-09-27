#![forbid(unsafe_code)]

//! EWW-K4e structural utilization of FLAT's packed-u16 fixed uniform envelope.
//!
//! This uses only the already-qualified host shadow constants. It does not
//! execute WGSL and does not claim physical allocation or bandwidth savings.

use flat_attention::{WGSL_PAGED_U16_PACKED_WORDS, WGSL_PAGED_U16_UNIFORM_U32};
use std::process::ExitCode;

const SCHEMA: &str = "kvlab.eww-k4e-packed-u16-uniform-utilization/v1";
const HEADER_U32: usize = 12;
const PAGES_PER_PACKED_U32: usize = 2;
const PAGE_COUNTS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const BASIS_POINTS: usize = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Utilization {
    mapped_pages: usize,
    useful_page_words: usize,
    useful_uniform_words: usize,
    fixed_uniform_words: usize,
    unused_uniform_words: usize,
    utilization_basis_points: usize,
}

fn utilization(mapped_pages: usize) -> Result<Utilization, String> {
    if mapped_pages == 0 || mapped_pages > WGSL_PAGED_U16_PACKED_WORDS * PAGES_PER_PACKED_U32 {
        return Err(format!(
            "mapped_pages must be in 1..={}, observed {mapped_pages}",
            WGSL_PAGED_U16_PACKED_WORDS * PAGES_PER_PACKED_U32
        ));
    }

    let useful_page_words = mapped_pages.div_ceil(PAGES_PER_PACKED_U32);
    let useful_uniform_words = HEADER_U32
        .checked_add(useful_page_words)
        .ok_or_else(|| "useful uniform word count overflow".to_owned())?;
    let fixed_uniform_words = WGSL_PAGED_U16_UNIFORM_U32;
    if useful_uniform_words > fixed_uniform_words {
        return Err("useful packed-u16 words exceed fixed uniform envelope".to_owned());
    }
    let unused_uniform_words = fixed_uniform_words - useful_uniform_words;
    let utilization_basis_points = useful_uniform_words
        .checked_mul(BASIS_POINTS)
        .map(|scaled| scaled / fixed_uniform_words)
        .ok_or_else(|| "utilization basis-point overflow".to_owned())?;

    Ok(Utilization {
        mapped_pages,
        useful_page_words,
        useful_uniform_words,
        fixed_uniform_words,
        unused_uniform_words,
        utilization_basis_points,
    })
}

fn run() -> Result<(), String> {
    println!(
        "schema,mapped_pages,useful_page_words,useful_uniform_words,fixed_uniform_words,useful_bytes,fixed_bytes,unused_bytes,utilization_basis_points,shader_status"
    );

    for mapped_pages in PAGE_COUNTS {
        let row = utilization(mapped_pages)?;
        let useful_bytes = row.useful_uniform_words * core::mem::size_of::<u32>();
        let fixed_bytes = row.fixed_uniform_words * core::mem::size_of::<u32>();
        let unused_bytes = row.unused_uniform_words * core::mem::size_of::<u32>();
        println!(
            "{SCHEMA},{},{},{},{},{useful_bytes},{fixed_bytes},{unused_bytes},{},shadow-only",
            row.mapped_pages,
            row.useful_page_words,
            row.useful_uniform_words,
            row.fixed_uniform_words,
            row.utilization_basis_points
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
    fn fixed_envelope_is_140_u32_or_560_bytes() {
        assert_eq!(WGSL_PAGED_U16_PACKED_WORDS, 128);
        assert_eq!(WGSL_PAGED_U16_UNIFORM_U32, 140);
        assert_eq!(WGSL_PAGED_U16_UNIFORM_U32 * core::mem::size_of::<u32>(), 560);
    }

    #[test]
    fn one_page_uses_one_packed_word_and_twelve_header_words() {
        let row = utilization(1).unwrap();
        assert_eq!(row.useful_page_words, 1);
        assert_eq!(row.useful_uniform_words, 13);
        assert_eq!(row.unused_uniform_words, 127);
    }

    #[test]
    fn full_256_page_domain_uses_entire_fixed_envelope() {
        let row = utilization(256).unwrap();
        assert_eq!(row.useful_page_words, 128);
        assert_eq!(row.useful_uniform_words, 140);
        assert_eq!(row.unused_uniform_words, 0);
        assert_eq!(row.utilization_basis_points, BASIS_POINTS);
    }

    #[test]
    fn utilization_is_monotone_across_frozen_sweep() {
        let mut previous = 0;
        for pages in PAGE_COUNTS {
            let current = utilization(pages).unwrap().utilization_basis_points;
            assert!(current >= previous);
            previous = current;
        }
    }

    #[test]
    fn zero_and_above_portable_domain_fail_closed() {
        assert!(utilization(0).is_err());
        assert!(utilization(257).is_err());
    }
}
