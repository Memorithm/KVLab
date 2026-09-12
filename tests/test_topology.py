import unittest

from kvlab.topology import (
    AttentionTopology,
    KvTopology,
    account_logical_kv_storage,
)


class TopologyAccountingTests(unittest.TestCase):
    def test_classifies_native_topologies(self) -> None:
        self.assertIs(KvTopology(8, 8, 2, 64, 2).topology, AttentionTopology.MHA)
        self.assertIs(KvTopology(8, 2, 2, 64, 2).topology, AttentionTopology.GQA)
        self.assertIs(KvTopology(8, 1, 2, 64, 2).topology, AttentionTopology.MQA)

    def test_gqa_accounts_only_declared_kv_heads(self) -> None:
        topology = KvTopology(
            query_heads=8,
            kv_heads=2,
            layers=4,
            head_dim=16,
            bytes_per_element=2,
        )
        result = account_logical_kv_storage(
            batch_size=2,
            token_count=10,
            topology=topology,
        )

        expected = 2 * 10 * 4 * 2 * 16 * 2 * 2
        expected_mha = 2 * 10 * 4 * 8 * 16 * 2 * 2
        self.assertEqual(result.logical_kv_bytes, expected)
        self.assertEqual(result.mha_reference_bytes, expected_mha)
        self.assertEqual(result.logical_fraction_of_mha, 0.25)
        self.assertEqual(result.logical_bytes_avoided_vs_mha, expected_mha - expected)
        self.assertEqual(topology.query_heads_per_kv_head, 4)

    def test_mqa_fraction_is_inverse_query_head_count(self) -> None:
        topology = KvTopology(16, 1, 1, 128, 2)
        result = account_logical_kv_storage(
            batch_size=1,
            token_count=32,
            topology=topology,
        )
        self.assertAlmostEqual(result.logical_fraction_of_mha, 1 / 16)

    def test_empty_batch_or_context_has_zero_storage(self) -> None:
        topology = KvTopology(4, 4, 2, 32, 2)
        result = account_logical_kv_storage(
            batch_size=0,
            token_count=100,
            topology=topology,
        )
        self.assertEqual(result.logical_kv_bytes, 0)
        self.assertEqual(result.mha_reference_bytes, 0)
        self.assertEqual(result.logical_fraction_of_mha, 0.0)

    def test_rejects_non_native_grouping_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            KvTopology(12, 5, 2, 64, 2)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            KvTopology(4, 8, 2, 64, 2)


if __name__ == "__main__":
    unittest.main()
