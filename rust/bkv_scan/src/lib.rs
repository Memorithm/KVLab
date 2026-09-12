#![forbid(unsafe_code)]

//! Exact BKV-K3 packed CPU scanner.
//!
//! This crate is a correctness backend for the Python BKV-K2/BKV-K3 oracle.
//! It makes no SIMD, multicore, NUMA, latency, throughput, or speedup claim.

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

pub fn scan_packed_pages(
    signature_bits: usize,
    query: &[u64],
    pages: &[Vec<u64>],
    max_distance: usize,
) -> Result<ScanResult, ScanError> {
    validate_words(signature_bits, query)?;
    if pages.is_empty() {
        return Err(ScanError::EmptyPages);
    }
    if max_distance > signature_bits {
        return Err(ScanError::DistanceOutOfRange);
    }

    let mut selected_pages = Vec::new();
    for (page_id, page) in pages.iter().enumerate() {
        validate_words(signature_bits, page)?;
        if hamming_distance(signature_bits, query, page)? as usize <= max_distance {
            selected_pages.push(page_id);
        }
    }

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
