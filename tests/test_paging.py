import unittest

from kvlab.paging import account_paged_kv, account_prefix_page_reuse


class PagedKvAccountingTests(unittest.TestCase):
    def test_exact_multiple_has_no_fragmentation(self) -> None:
        result = account_paged_kv(
            token_count=32, bytes_per_token=128, page_size_tokens=16
        )
        self.assertEqual(len(result.pages), 2)
        self.assertEqual(result.logical_kv_bytes, 4096)
        self.assertEqual(result.allocated_bytes, 4096)
        self.assertEqual(result.fragmentation_bytes, 0)

    def test_partial_final_page_tracks_internal_fragmentation(self) -> None:
        result = account_paged_kv(
            token_count=17, bytes_per_token=100, page_size_tokens=16
        )
        self.assertEqual([page.token_count for page in result.pages], [16, 1])
        self.assertEqual(result.logical_kv_bytes, 1700)
        self.assertEqual(result.allocated_bytes, 3200)
        self.assertEqual(result.fragmentation_bytes, 1500)

    def test_empty_cache_uses_no_pages(self) -> None:
        result = account_paged_kv(
            token_count=0, bytes_per_token=0, page_size_tokens=16
        )
        self.assertEqual(result.pages, ())
        self.assertEqual(result.logical_kv_bytes, 0)
        self.assertEqual(result.allocated_bytes, 0)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            account_paged_kv(token_count=-1, bytes_per_token=1, page_size_tokens=16)
        with self.assertRaises(ValueError):
            account_paged_kv(token_count=1, bytes_per_token=0, page_size_tokens=16)
        with self.assertRaises(ValueError):
            account_paged_kv(token_count=1, bytes_per_token=1, page_size_tokens=0)


class PrefixReuseAccountingTests(unittest.TestCase):
    def test_sharing_complete_prefix_pages_reduces_duplication_not_logical_size(self) -> None:
        result = account_prefix_page_reuse(
            request_count=4,
            prefix_tokens=32,
            bytes_per_token=64,
            page_size_tokens=16,
        )
        self.assertEqual(result.shared_full_pages, 2)
        self.assertEqual(result.logical_kv_bytes, 4 * 32 * 64)
        self.assertEqual(result.physical_bytes_without_sharing, 4 * 32 * 64)
        self.assertEqual(result.physical_bytes_with_sharing, 32 * 64)
        self.assertEqual(result.duplicate_bytes_avoided, 3 * 32 * 64)

    def test_partial_prefix_page_is_not_shared_in_conservative_model(self) -> None:
        result = account_prefix_page_reuse(
            request_count=3,
            prefix_tokens=17,
            bytes_per_token=10,
            page_size_tokens=16,
        )
        self.assertEqual(result.shared_full_pages, 1)
        self.assertEqual(result.logical_kv_bytes, 3 * 170)
        self.assertEqual(result.physical_bytes_without_sharing, 3 * 320)
        self.assertEqual(result.physical_bytes_with_sharing, 640)
        self.assertEqual(result.duplicate_bytes_avoided, 320)

    def test_single_request_has_no_duplicate_savings(self) -> None:
        result = account_prefix_page_reuse(
            request_count=1,
            prefix_tokens=64,
            bytes_per_token=2,
            page_size_tokens=16,
        )
        self.assertEqual(result.duplicate_bytes_avoided, 0)
        self.assertEqual(
            result.physical_bytes_with_sharing, result.physical_bytes_without_sharing
        )

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            account_prefix_page_reuse(
                request_count=0,
                prefix_tokens=1,
                bytes_per_token=1,
                page_size_tokens=16,
            )
        with self.assertRaises(ValueError):
            account_prefix_page_reuse(
                request_count=1,
                prefix_tokens=1,
                bytes_per_token=0,
                page_size_tokens=16,
            )


if __name__ == "__main__":
    unittest.main()
