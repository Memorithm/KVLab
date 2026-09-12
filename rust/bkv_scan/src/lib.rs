#![forbid(unsafe_code)]

//! Exact BKV-K3 packed CPU scanners.
//!
//! The scalar scanner is the correctness oracle for the parallel backend. The
//! parallel implementation uses deterministic contiguous shards and merges
//! candidates in page order. This crate makes no SIMD, NUMA, latency,
//! throughput, scaling, or speedup claim.

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
