import pytest

from kvlab.topology import (
    AttentionTopology,
    KvTopology,
    account_logical_kv_storage,
)


def test_classifies_native_topologies() -> None:
    assert KvTopology(8, 8, 2, 64, 2).topology is AttentionTopology.MHA
    assert KvTopology(8, 2, 2, 64, 2).topology is AttentionTopology.GQA
    assert KvTopology(8, 1, 2, 64, 2).topology is AttentionTopology.MQA


def test_gqa_accounts_only_declared_kv_heads() -> None:
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
    assert result.logical_kv_bytes == expected
    assert result.mha_reference_bytes == expected_mha
    assert result.logical_fraction_of_mha == 0.25
    assert result.logical_bytes_avoided_vs_mha == expected_mha - expected
    assert topology.query_heads_per_kv_head == 4


def test_mqa_fraction_is_inverse_query_head_count() -> None:
    topology = KvTopology(16, 1, 1, 128, 2)
    result = account_logical_kv_storage(
        batch_size=1,
        token_count=32,
        topology=topology,
    )
    assert result.logical_fraction_of_mha == pytest.approx(1 / 16)


def test_empty_batch_or_context_has_zero_storage() -> None:
    topology = KvTopology(4, 4, 2, 32, 2)
    result = account_logical_kv_storage(
        batch_size=0,
        token_count=100,
        topology=topology,
    )
    assert result.logical_kv_bytes == 0
    assert result.mha_reference_bytes == 0
    assert result.logical_fraction_of_mha == 0.0


def test_rejects_non_native_grouping_geometry() -> None:
    with pytest.raises(ValueError, match="divisible"):
        KvTopology(12, 5, 2, 64, 2)
    with pytest.raises(ValueError, match="cannot exceed"):
        KvTopology(4, 8, 2, 64, 2)
