import unittest

from kvlab.boolean_kv import (
    BooleanKvCache,
    BooleanKvError,
    PackedBits,
    pack_bits,
)


class PackedBitsTests(unittest.TestCase):
    def test_pack_is_canonical_across_word_boundary(self) -> None:
        bits = [False] * 65
        bits[0] = True
        bits[63] = True
        bits[64] = True
        packed = pack_bits(bits)
        self.assertEqual(packed.bit_length, 65)
        self.assertEqual(packed.words, ((1 << 63) | 1, 1))
        self.assertEqual(packed.logical_bits, 65)
        self.assertEqual(packed.physical_bits, 128)
        self.assertEqual(packed.physical_bytes, 16)

    def test_tail_bits_must_be_zero(self) -> None:
        with self.assertRaises(BooleanKvError):
            PackedBits(bit_length=65, words=(0, 2))

    def test_hamming_and_xnor_are_exact(self) -> None:
        left = pack_bits([False, True, False, True, True])
        right = pack_bits([False, False, False, True, False])
        self.assertEqual(left.hamming_distance(right), 2)
        self.assertEqual(left.xnor_matches(right), 3)

    def test_width_mismatch_fails_closed(self) -> None:
        with self.assertRaises(BooleanKvError):
            pack_bits([True]).hamming_distance(pack_bits([True, False]))


class BooleanKvCacheTests(unittest.TestCase):
    def test_accounting_distinguishes_logical_and_physical_storage(self) -> None:
        cache = BooleanKvCache(signature_bits=65)
        key = pack_bits([False] * 65)
        value = pack_bits([True] + [False] * 64)
        cache.append(key, value)
        cache.append(value)
        accounting = cache.accounting()
        self.assertEqual(accounting.pages, 2)
        self.assertEqual(accounting.key_logical_bits, 130)
        self.assertEqual(accounting.value_logical_bits, 65)
        self.assertEqual(accounting.key_physical_bytes, 32)
        self.assertEqual(accounting.value_physical_bytes, 16)
        self.assertEqual(accounting.total_physical_bytes, 48)

    def test_reset_invalidates_old_generation_and_reuses_logical_page_zero(self) -> None:
        cache = BooleanKvCache(signature_bits=4)
        old = cache.append(pack_bits([True, False, False, False]))
        self.assertEqual(old.logical_page, 0)
        self.assertEqual(old.generation, 0)
        cache.reset()
        self.assertEqual(cache.generation, 1)
        self.assertIsNone(cache.page(0))
        new = cache.append(pack_bits([False, True, False, False]))
        self.assertEqual(new.logical_page, 0)
        self.assertEqual(new.generation, 1)

    def test_search_is_deterministic_and_distance_ordered(self) -> None:
        cache = BooleanKvCache(signature_bits=4)
        cache.append(pack_bits([False, False, False, False]))
        cache.append(pack_bits([True, False, False, False]))
        cache.append(pack_bits([False, True, False, False]))
        cache.append(pack_bits([True, True, True, True]))

        matches = cache.search_hamming(
            pack_bits([False, False, False, False]),
            max_distance=1,
        )
        self.assertEqual(
            [(item.logical_page, item.hamming_distance, item.xnor_matches) for item in matches],
            [(0, 0, 4), (1, 1, 3), (2, 1, 3)],
        )

        limited = cache.search_hamming(
            pack_bits([False, False, False, False]),
            max_distance=4,
            limit=2,
        )
        self.assertEqual([item.logical_page for item in limited], [0, 1])

    def test_append_rejects_incompatible_widths(self) -> None:
        cache = BooleanKvCache(signature_bits=8)
        with self.assertRaises(BooleanKvError):
            cache.append(pack_bits([False] * 7))

    def test_value_signature_width_must_match_key(self) -> None:
        cache = BooleanKvCache(signature_bits=4)
        with self.assertRaises(BooleanKvError):
            cache.append(
                pack_bits([False] * 4),
                pack_bits([False] * 3),
            )

    def test_search_validation_fails_closed(self) -> None:
        cache = BooleanKvCache(signature_bits=4)
        cache.append(pack_bits([False] * 4))
        with self.assertRaises(BooleanKvError):
            cache.search_hamming(pack_bits([False] * 4), max_distance=5)
        with self.assertRaises(BooleanKvError):
            cache.search_hamming(pack_bits([False] * 4), max_distance=1, limit=0)


if __name__ == "__main__":
    unittest.main()
