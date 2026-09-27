#![forbid(unsafe_code)]

//! EWW-K4 host qualification against FLAT's production W64 paged-KV table.
//!
//! This measures the real production control-plane representation pinned at the
//! merge that promoted epoch-scoped generation plus one u64 physical-page lane
//! per mapping. It does not execute numerical attention or claim GPU/model
//! performance.

use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use std::env;
use std::hint::black_box;
use std::process::ExitCode;
use std::time::Instant;

const SCHEMA: &str = "kvlab.eww-k4-flat-production-w64/v1";
const FLAT_REVISION: &str = "ae61403a4b4b4ed93ea8fa1bb4cd632be9038571";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Config {
    page_size: usize,
    physical_pages: usize,
    live_tokens: usize,
    warmup: usize,
    repetitions: usize,
}

fn parse_usize(name: &str, value: Option<String>) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("missing {name}"))?
        .parse::<usize>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn parse_config_from<I>(args: I) -> Result<Config, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let config = Config {
        page_size: parse_usize("page_size", args.next())?,
        physical_pages: parse_usize("physical_pages", args.next())?,
        live_tokens: parse_usize("live_tokens", args.next())?,
        warmup: parse_usize("warmup", args.next())?,
        repetitions: parse_usize("repetitions", args.next())?,
    };
    if args.next().is_some() {
        return Err("too many arguments".to_owned());
    }
    if config.repetitions == 0 {
        return Err("repetitions must be non-zero".to_owned());
    }
    Ok(config)
}

fn parse_config() -> Result<Config, String> {
    parse_config_from(env::args().skip(1))
}

fn build_table(config: Config) -> Result<PagedKvTable, String> {
    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: config.page_size,
        physical_pages: config.physical_pages,
    })
    .map_err(|error| error.to_string())?;
    table
        .append(config.live_tokens)
        .map_err(|error| error.to_string())?;
    Ok(table)
}

fn verify_production_w64(table: &PagedKvTable) -> Result<(), String> {
    let config = table.config();
    let expected_pages = if table.is_empty() {
        0
    } else {
        table.len().div_ceil(config.page_size)
    };
    if table.mapped_page_lanes().len() != expected_pages {
        return Err(format!(
            "W64 lane count mismatch: observed {}, expected {expected_pages}",
            table.mapped_page_lanes().len()
        ));
    }

    for (logical_page, physical_lane) in table.mapped_page_lanes().iter().copied().enumerate() {
        let expected = u64::try_from(logical_page)
            .map_err(|_| "logical page index does not fit u64".to_owned())?;
        if physical_lane != expected {
            return Err(format!(
                "deterministic page map drift at logical page {logical_page}: {physical_lane} != {expected}"
            ));
        }
    }

    for logical_token in 0..table.len() {
        let address = table
            .address(logical_token)
            .ok_or_else(|| format!("missing address for live token {logical_token}"))?;
        let expected_page = logical_token / config.page_size;
        let expected_offset = logical_token % config.page_size;
        if address.physical_page != expected_page
            || address.offset_in_page != expected_offset
            || address.generation != table.generation()
        {
            return Err(format!("address parity drift at token {logical_token}"));
        }
    }

    if table.address(table.len()).is_some() {
        return Err("first out-of-range token unexpectedly resolved".to_owned());
    }
    Ok(())
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
        checksum = checksum.rotate_left(9) ^ page.rotate_left(3) ^ offset ^ address.generation;
    }
    Ok(checksum)
}

fn lane_checksum(table: &PagedKvTable) -> u64 {
    table
        .mapped_page_lanes()
        .iter()
        .copied()
        .fold(table.generation(), |acc, lane| acc.rotate_left(7) ^ lane)
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

fn time_lookup(
    table: &PagedKvTable,
    warmup: usize,
    repetitions: usize,
) -> Result<(u128, u128, u128, u64), String> {
    for _ in 0..warmup {
        black_box(lookup_checksum(table)?);
    }
    let mut samples = Vec::with_capacity(repetitions);
    let mut checksum = 0_u64;
    for _ in 0..repetitions {
        let start = Instant::now();
        checksum = black_box(lookup_checksum(table)?);
        samples.push(start.elapsed().as_nanos());
    }
    let min = *samples.iter().min().expect("non-empty repetitions");
    let max = *samples.iter().max().expect("non-empty repetitions");
    let med = median(&mut samples);
    Ok((med, min, max, checksum))
}

fn time_lane_walk(
    table: &PagedKvTable,
    warmup: usize,
    repetitions: usize,
) -> (u128, u128, u128, u64) {
    for _ in 0..warmup {
        black_box(lane_checksum(table));
    }
    let mut samples = Vec::with_capacity(repetitions);
    let mut checksum = 0_u64;
    for _ in 0..repetitions {
        let start = Instant::now();
        checksum = black_box(lane_checksum(table));
        samples.push(start.elapsed().as_nanos());
    }
    let min = *samples.iter().min().expect("non-empty repetitions");
    let max = *samples.iter().max().expect("non-empty repetitions");
    let med = median(&mut samples);
    (med, min, max, checksum)
}

fn run(config: Config) -> Result<(), String> {
    let table = build_table(config)?;
    verify_production_w64(&table)?;

    let mapped_pages = table.mapped_page_lanes().len();
    let page_map_payload_bytes = mapped_pages
        .checked_mul(core::mem::size_of::<u64>())
        .ok_or_else(|| "page-map byte count overflow".to_owned())?;
    let epoch_generation_payload_bytes = core::mem::size_of::<u64>();
    let declared_control_payload_bytes = page_map_payload_bytes
        .checked_add(epoch_generation_payload_bytes)
        .ok_or_else(|| "control payload byte count overflow".to_owned())?;
    let prior_w128_shadow_payload_bytes = mapped_pages
        .checked_mul(2 * core::mem::size_of::<u64>())
        .and_then(|bytes| bytes.checked_add(epoch_generation_payload_bytes))
        .ok_or_else(|| "W128 shadow payload byte count overflow".to_owned())?;

    let (lookup_median, lookup_min, lookup_max, lookup_checksum) =
        time_lookup(&table, config.warmup, config.repetitions)?;
    let (lane_median, lane_min, lane_max, lane_checksum) =
        time_lane_walk(&table, config.warmup, config.repetitions);

    println!(
        "schema,flat_revision,page_size,physical_pages,live_tokens,mapped_pages,generation,page_map_payload_bytes,epoch_generation_payload_bytes,declared_control_payload_bytes,prior_w128_shadow_payload_bytes,lookup_median_ns,lookup_min_ns,lookup_max_ns,lane_walk_median_ns,lane_walk_min_ns,lane_walk_max_ns,lookup_checksum,lane_checksum,kv_bytes_touched_status,ttft_status,tpot_status,tokens_per_second_status"
    );
    println!(
        "{SCHEMA},{FLAT_REVISION},{},{},{},{mapped_pages},{},{page_map_payload_bytes},{epoch_generation_payload_bytes},{declared_control_payload_bytes},{prior_w128_shadow_payload_bytes},{lookup_median},{lookup_min},{lookup_max},{lane_median},{lane_min},{lane_max},{lookup_checksum},{lane_checksum},not-measured,not-measured,not-measured,not-measured",
        config.page_size,
        config.physical_pages,
        config.live_tokens,
        table.generation(),
    );

    Ok(())
}

fn main() -> ExitCode {
    match parse_config().and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            eprintln!(
                "usage: eww_flat_production_w64_bench <page_size> <physical_pages> <live_tokens> <warmup> <repetitions>"
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_w64_payload_accounting_is_exact_for_mapped_pages() {
        let config = Config {
            page_size: 4,
            physical_pages: 8,
            live_tokens: 17,
            warmup: 0,
            repetitions: 1,
        };
        let table = build_table(config).unwrap();
        verify_production_w64(&table).unwrap();
        assert_eq!(table.mapped_page_lanes().len(), 5);
        assert_eq!(
            core::mem::size_of_val(table.mapped_page_lanes()),
            5 * core::mem::size_of::<u64>()
        );
    }

    #[test]
    fn reset_moves_generation_without_duplicating_it_in_page_lanes() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 2,
            physical_pages: 4,
        })
        .unwrap();
        table.append(5).unwrap();
        let before = table.mapped_page_lanes().to_vec();
        table.reset().unwrap();
        table.append(5).unwrap();

        assert_eq!(table.generation(), 1);
        assert_eq!(table.mapped_page_lanes(), before);
        assert!(table
            .mapped_page_metadata()
            .all(|(_, generation)| generation == 1));
    }

    #[test]
    fn truncate_and_reappend_preserve_deterministic_w64_mapping() {
        let mut table = PagedKvTable::new(PagedKvConfig {
            page_size: 4,
            physical_pages: 5,
        })
        .unwrap();
        table.append(17).unwrap();
        table.truncate(5).unwrap();
        table.append(8).unwrap();

        assert_eq!(table.mapped_page_lanes(), &[0, 1, 2, 3]);
        verify_production_w64(&table).unwrap();
    }

    #[test]
    fn parser_rejects_zero_repetitions() {
        let error = parse_config_from([
            "4".to_owned(),
            "8".to_owned(),
            "17".to_owned(),
            "2".to_owned(),
            "0".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("repetitions"));
    }
}
