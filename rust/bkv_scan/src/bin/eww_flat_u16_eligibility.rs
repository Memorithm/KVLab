#![forbid(unsafe_code)]

//! EWW-K4d exact eligibility boundaries for FLAT's packed-u16 page-map shadow.
//!
//! This is a structural qualification only. It does not execute WGSL or claim
//! a performance benefit.

use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::{
    PagedDecodeError, WgpuPackedPagedKvTable16, WGSL_PAGED_U16_MAX_PHYSICAL_PAGES,
};
use std::process::ExitCode;

const SCHEMA: &str = "kvlab.eww-k4d-packed-u16-eligibility/v1";
const PORTABLE_LOGICAL_PAGE_LIMIT: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Case {
    name: &'static str,
    physical_pages: usize,
    mapped_pages: usize,
    expected: Expected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Expected {
    Eligible,
    TooManyLogicalPages,
    PhysicalPageDomainTooWide,
}

const CASES: [Case; 6] = [
    Case {
        name: "single-page",
        physical_pages: 1,
        mapped_pages: 1,
        expected: Expected::Eligible,
    },
    Case {
        name: "u16-domain-minus-one",
        physical_pages: WGSL_PAGED_U16_MAX_PHYSICAL_PAGES - 1,
        mapped_pages: 1,
        expected: Expected::Eligible,
    },
    Case {
        name: "u16-domain-exact",
        physical_pages: WGSL_PAGED_U16_MAX_PHYSICAL_PAGES,
        mapped_pages: 1,
        expected: Expected::Eligible,
    },
    Case {
        name: "u16-domain-plus-one",
        physical_pages: WGSL_PAGED_U16_MAX_PHYSICAL_PAGES + 1,
        mapped_pages: 1,
        expected: Expected::PhysicalPageDomainTooWide,
    },
    Case {
        name: "portable-logical-limit",
        physical_pages: PORTABLE_LOGICAL_PAGE_LIMIT,
        mapped_pages: PORTABLE_LOGICAL_PAGE_LIMIT,
        expected: Expected::Eligible,
    },
    Case {
        name: "portable-logical-limit-plus-one",
        physical_pages: PORTABLE_LOGICAL_PAGE_LIMIT + 1,
        mapped_pages: PORTABLE_LOGICAL_PAGE_LIMIT + 1,
        expected: Expected::TooManyLogicalPages,
    },
];

fn build(case: Case) -> Result<PagedKvTable, String> {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 1,
        physical_pages: case.physical_pages,
    })
    .map_err(|error| error.to_string())?;
    table
        .append(case.mapped_pages)
        .map_err(|error| error.to_string())?;
    Ok(table)
}

fn classify(case: Case) -> Result<Expected, String> {
    let table = build(case)?;
    match WgpuPackedPagedKvTable16::from_table(&table) {
        Ok(packed) => {
            if packed.mapped_pages() != case.mapped_pages {
                return Err(format!(
                    "{}: mapped page drift {} != {}",
                    case.name,
                    packed.mapped_pages(),
                    case.mapped_pages
                ));
            }
            Ok(Expected::Eligible)
        }
        Err(PagedDecodeError::TooManyMappedPages { actual, maximum }) => {
            if actual != case.mapped_pages || maximum != PORTABLE_LOGICAL_PAGE_LIMIT {
                return Err(format!(
                    "{}: unexpected logical limit payload actual={actual} maximum={maximum}",
                    case.name
                ));
            }
            Ok(Expected::TooManyLogicalPages)
        }
        Err(PagedDecodeError::PhysicalPageIndexExceedsU16 { physical_page }) => {
            let expected_last = u64::try_from(case.physical_pages - 1)
                .map_err(|_| "physical page domain does not fit u64".to_owned())?;
            if physical_page != expected_last {
                return Err(format!(
                    "{}: physical domain diagnostic {physical_page} != {expected_last}",
                    case.name
                ));
            }
            Ok(Expected::PhysicalPageDomainTooWide)
        }
        Err(error) => Err(format!("{}: unexpected error {error}", case.name)),
    }
}

fn run() -> Result<(), String> {
    println!("schema,case,physical_pages,mapped_pages,expected,observed,eligible");
    for case in CASES {
        let observed = classify(case)?;
        if observed != case.expected {
            return Err(format!(
                "{}: eligibility mismatch {:?} != {:?}",
                case.name, observed, case.expected
            ));
        }
        println!(
            "{SCHEMA},{},{},{},{:?},{:?},{}",
            case.name,
            case.physical_pages,
            case.mapped_pages,
            case.expected,
            observed,
            observed == Expected::Eligible
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
    fn frozen_boundary_matrix_matches_contract() {
        for case in CASES {
            assert_eq!(classify(case).unwrap(), case.expected, "{}", case.name);
        }
    }

    #[test]
    fn exact_u16_domain_is_65536_physical_pages() {
        assert_eq!(WGSL_PAGED_U16_MAX_PHYSICAL_PAGES, 65_536);
    }

    #[test]
    fn portable_logical_page_limit_remains_256() {
        assert_eq!(PORTABLE_LOGICAL_PAGE_LIMIT, 256);
    }
}
