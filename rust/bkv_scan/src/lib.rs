#![forbid(unsafe_code)]

//! Exact BKV-K3 packed CPU scanners.
//!
//! The scalar scanner is the correctness oracle for the parallel backend. The
//! parallel implementation uses deterministic contiguous shards and merges
//! candidates in page order. This crate makes no SIMD, NUMA, latency,
//! throughput, scaling, or speedup claim.

pub mod cps2_compact_quality;
pub mod provenance;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanResult {
    pub selected_pages: Vec<usize>,
    pub pages_scanned: usize,
    pub signature_bits: usize,
    pub bits_compared: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanError {
    ZeroSignatureBits,
    EmptyPages,
    InvalidWordCount { expected: usize, actual: usize },
    FlatStorageLengthMismatch { expected: usize, actual: usize },
    StorageLengthOverflow,
    NonZeroTailBits,
    DistanceOutOfRange,
    ZeroWorkers,
    WorkerPanicked,
}

fn word_count(signature_bits: usize) -> Result<usize, ScanError> {
    if signature_bits == 0 {
        return Err(ScanError::ZeroSignatureBits);
    }
    Ok(signature_bits.div_ceil(64))
}

fn validate_words(signature_bits: usize, words: &[u64]) -> Result<(), ScanError> {
    let expected = word_count(signature_bits)?;
    if words.len() != expected {
        return Err(ScanError::InvalidWordCount {
            expected,
            actual: words.len(),
        });
    }

    let tail_bits = signature_bits % 64;
    if tail_bits != 0 {
        let valid_mask = (1_u64 << tail_bits) - 1;
        if words[words.len() - 1] & !valid_mask != 0 {
            return Err(ScanError::NonZeroTailBits);
        }
    }
    Ok(())
}

fn validate_scan_inputs(
    signature_bits: usize,
    query: &[u64],
    pages: &[Vec<u64>],
    max_distance: usize,
) -> Result<(), ScanError> {
    validate_words(signature_bits, query)?;
    if pages.is_empty() {
        return Err(ScanError::EmptyPages);
    }
    if max_distance > signature_bits {
        return Err(ScanError::DistanceOutOfRange);
    }
    for page in pages {
        validate_words(signature_bits, page)?;
    }
    Ok(())
}

pub fn hamming_distance(
    signature_bits: usize,
    left: &[u64],
    right: &[u64],
) -> Result<u32, ScanError> {
    validate_words(signature_bits, left)?;
    validate_words(signature_bits, right)?;
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum())
}

fn hamming_distance_validated(left: &[u64], right: &[u64]) -> u32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum()
}

pub fn scan_packed_pages(
    signature_bits: usize,
    query: &[u64],
    pages: &[Vec<u64>],
    max_distance: usize,
) -> Result<ScanResult, ScanError> {
    validate_scan_inputs(signature_bits, query, pages, max_distance)?;

    let selected_pages = pages
        .iter()
        .enumerate()
        .filter_map(|(page_id, page)| {
            (hamming_distance_validated(query, page) as usize <= max_distance).then_some(page_id)
        })
        .collect();

    Ok(ScanResult {
        selected_pages,
        pages_scanned: pages.len(),
        signature_bits,
        bits_compared: pages.len() * signature_bits,
    })
}

/// Scan exact packed signatures stored as one contiguous `u64` plane.
///
/// The signature width is carried once for the complete plane. Pages are laid
/// out consecutively with no per-page `Vec`, enum or width tag:
///
/// ```text
/// [page0 lanes...][page1 lanes...][page2 lanes...]...
/// ```
///
/// This is a representation/validation path only; it makes no locality,
/// allocation, cache-hit, latency or throughput claim.
pub fn scan_packed_pages_flat(
    signature_bits: usize,
    query: &[u64],
    page_count: usize,
    flat_pages: &[u64],
    max_distance: usize,
) -> Result<ScanResult, ScanError> {
    validate_words(signature_bits, query)?;
    if page_count == 0 {
        return Err(ScanError::EmptyPages);
    }
    if max_distance > signature_bits {
        return Err(ScanError::DistanceOutOfRange);
    }

    let words_per_page = word_count(signature_bits)?;
    let expected_lanes = page_count
        .checked_mul(words_per_page)
        .ok_or(ScanError::StorageLengthOverflow)?;
    if flat_pages.len() != expected_lanes {
        return Err(ScanError::FlatStorageLengthMismatch {
            expected: expected_lanes,
            actual: flat_pages.len(),
        });
    }

    let tail_bits = signature_bits % 64;
    let tail_mask = (tail_bits != 0).then(|| (1_u64 << tail_bits) - 1);
    let mut selected_pages = Vec::new();

    for (page_id, page) in flat_pages.chunks_exact(words_per_page).enumerate() {
        if let Some(valid_mask) = tail_mask {
            if page[words_per_page - 1] & !valid_mask != 0 {
                return Err(ScanError::NonZeroTailBits);
            }
        }

        if hamming_distance_validated(query, page) as usize <= max_distance {
            selected_pages.push(page_id);
        }
    }

    Ok(ScanResult {
        selected_pages,
        pages_scanned: page_count,
        signature_bits,
        bits_compared: page_count
            .checked_mul(signature_bits)
            .ok_or(ScanError::StorageLengthOverflow)?,
    })
}

/// Scan exact packed signatures on scoped worker threads.
///
/// The page set is split into deterministic contiguous shards. Results are
/// joined in shard order, so candidate ordering is identical to the scalar
/// oracle for every worker count.
pub fn scan_packed_pages_parallel(
    signature_bits: usize,
    query: &[u64],
    pages: &[Vec<u64>],
    max_distance: usize,
    workers: usize,
) -> Result<ScanResult, ScanError> {
    validate_scan_inputs(signature_bits, query, pages, max_distance)?;
    if workers == 0 {
        return Err(ScanError::ZeroWorkers);
    }

    let worker_count = workers.min(pages.len());
    let shard_size = pages.len().div_ceil(worker_count);
    let selected_pages = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        for shard_start in (0..pages.len()).step_by(shard_size) {
            let shard_end = (shard_start + shard_size).min(pages.len());
            let shard = &pages[shard_start..shard_end];
            handles.push(scope.spawn(move || {
                shard
                    .iter()
                    .enumerate()
                    .filter_map(|(local_id, page)| {
                        (hamming_distance_validated(query, page) as usize <= max_distance)
                            .then_some(shard_start + local_id)
                    })
                    .collect::<Vec<_>>()
            }));
        }

        let mut merged = Vec::new();
        for handle in handles {
            let mut shard_candidates = handle.join().map_err(|_| ScanError::WorkerPanicked)?;
            merged.append(&mut shard_candidates);
        }
        Ok::<Vec<usize>, ScanError>(merged)
    })?;

    Ok(ScanResult {
        selected_pages,
        pages_scanned: pages.len(),
        signature_bits,
        bits_compared: pages.len() * signature_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pages() -> Vec<Vec<u64>> {
        vec![
            vec![0b1010_u64],
            vec![0b1110_u64],
            vec![0b0000_u64],
            vec![0b1011_u64],
            vec![0b0010_u64],
            vec![0b1111_u64],
            vec![0b1000_u64],
        ]
    }

    #[test]
    fn exact_scan_matches_hand_computed_candidates() {
        let query = [0b1010_u64];
        let pages = vec![
            vec![0b1010_u64],
            vec![0b1110_u64],
            vec![0b0000_u64],
            vec![0b1011_u64],
        ];
        let result = scan_packed_pages(4, &query, &pages, 1).unwrap();
        assert_eq!(result.selected_pages, vec![0, 1, 3]);
        assert_eq!(result.pages_scanned, 4);
        assert_eq!(result.bits_compared, 16);
    }

    #[test]
    fn parallel_scan_matches_scalar_for_worker_sweep() {
        let query = [0b1010_u64];
        let pages = sample_pages();
        let scalar = scan_packed_pages(4, &query, &pages, 1).unwrap();
        for workers in [1, 2, 3, 4, 7, 16] {
            let parallel = scan_packed_pages_parallel(4, &query, &pages, 1, workers).unwrap();
            assert_eq!(parallel, scalar, "worker count {workers}");
        }
    }

    #[test]
    fn parallel_scan_preserves_page_order_across_uneven_shards() {
        let query = [0_u64];
        let pages = vec![vec![0], vec![1], vec![0], vec![3], vec![0]];
        let result = scan_packed_pages_parallel(2, &query, &pages, 0, 3).unwrap();
        assert_eq!(result.selected_pages, vec![0, 2, 4]);
    }

    #[test]
    fn parallel_scan_rejects_zero_workers() {
        let query = [0_u64];
        let pages = vec![vec![0_u64]];
        assert_eq!(
            scan_packed_pages_parallel(1, &query, &pages, 0, 0),
            Err(ScanError::ZeroWorkers)
        );
    }

    #[test]
    fn little_bit_order_words_match_python_contract() {
        let left = [1_u64 << 63, 1_u64];
        let right = [0_u64, 0_u64];
        assert_eq!(hamming_distance(65, &left, &right).unwrap(), 2);
    }

    #[test]
    fn rejects_non_zero_unused_tail_bits() {
        let query = [0b1_0000_u64];
        let pages = vec![vec![0_u64]];
        assert_eq!(
            scan_packed_pages(4, &query, &pages, 1),
            Err(ScanError::NonZeroTailBits)
        );
    }

    #[test]
    fn flat_plane_matches_nested_oracle_across_frozen_elastic_widths() {
        for signature_bits in [64_usize, 128, 256, 512, 1024, 2048] {
            let words_per_page = signature_bits / 64;
            let query = (0..words_per_page)
                .map(|index| {
                    0xd6e8_feb8_6659_fd93_u64.wrapping_mul((index as u64).wrapping_add(1))
                        ^ 0x94d0_49bb_1331_11eb
                })
                .collect::<Vec<_>>();

            let mut near = query.clone();
            near[0] ^= 1;
            let nested = vec![
                query.clone(),
                near,
                vec![0_u64; words_per_page],
                vec![u64::MAX; words_per_page],
            ];
            let flat = nested
                .iter()
                .flat_map(|page| page.iter().copied())
                .collect::<Vec<_>>();

            let oracle = scan_packed_pages(signature_bits, &query, &nested, 1).unwrap();
            let contiguous =
                scan_packed_pages_flat(signature_bits, &query, nested.len(), &flat, 1).unwrap();

            assert_eq!(contiguous, oracle, "width {signature_bits}");
        }
    }

    #[test]
    fn flat_plane_rejects_incomplete_or_extra_storage() {
        let query = [0_u64, 0_u64];

        assert_eq!(
            scan_packed_pages_flat(128, &query, 2, &[0, 0, 0], 0),
            Err(ScanError::FlatStorageLengthMismatch {
                expected: 4,
                actual: 3
            })
        );
        assert_eq!(
            scan_packed_pages_flat(128, &query, 1, &[0, 0, 0], 0),
            Err(ScanError::FlatStorageLengthMismatch {
                expected: 2,
                actual: 3
            })
        );
    }

    #[test]
    fn flat_plane_preserves_tail_validation() {
        let query = [0_u64];
        assert_eq!(
            scan_packed_pages_flat(4, &query, 1, &[0b1_0000], 1),
            Err(ScanError::NonZeroTailBits)
        );
    }

    #[test]
    fn flat_plane_rejects_zero_pages() {
        assert_eq!(
            scan_packed_pages_flat(64, &[0], 0, &[], 0),
            Err(ScanError::EmptyPages)
        );
    }

    #[test]
    fn frozen_elastic_widths_match_scalar_and_parallel_oracles() {
        for signature_bits in [64_usize, 128, 256, 512, 1024, 2048] {
            let word_count = signature_bits / 64;
            let query = (0..word_count)
                .map(|index| {
                    0x9e37_79b9_7f4a_7c15_u64.wrapping_mul((index as u64).wrapping_add(1))
                        ^ 0xa5a5_5a5a_0123_4567
                })
                .collect::<Vec<_>>();

            let mut one_bit = query.clone();
            one_bit[word_count - 1] ^= 1_u64 << ((word_count - 1) % 64);

            assert_eq!(
                hamming_distance(signature_bits, &query, &one_bit).unwrap(),
                1,
                "width {signature_bits}"
            );

            let pages = vec![
                query.clone(),
                one_bit,
                vec![0_u64; word_count],
                vec![u64::MAX; word_count],
            ];
            let scalar = scan_packed_pages(signature_bits, &query, &pages, 1).unwrap();

            for workers in [1_usize, 2, 4, 8] {
                let parallel =
                    scan_packed_pages_parallel(signature_bits, &query, &pages, 1, workers).unwrap();
                assert_eq!(
                    parallel, scalar,
                    "width {signature_bits}, worker count {workers}"
                );
            }
        }
    }

    #[test]
    fn rejects_page_width_mismatch() {
        let query = [0_u64];
        let pages = vec![vec![0_u64, 0_u64]];
        assert_eq!(
            scan_packed_pages(64, &query, &pages, 1),
            Err(ScanError::InvalidWordCount {
                expected: 1,
                actual: 2
            })
        );
    }
}
