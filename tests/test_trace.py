import pytest

from kvlab.trace import KvEventKind, KvResourceEffect, KvTrace, KvTraceEvent


def test_trace_requires_strict_sequence_order() -> None:
    event = KvTraceEvent(
        sequence=1,
        token_start=0,
        token_count=1,
        page_id=0,
        block_id=0,
        kind=KvEventKind.READ,
        resource_effect=KvResourceEffect.TRANSFER,
        bytes_observed=16,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        KvTrace((event, event))


def test_token_lookup_preserves_page_and_block_identity() -> None:
    first = KvTraceEvent(
        sequence=0,
        token_start=0,
        token_count=4,
        page_id=2,
        block_id=7,
        kind=KvEventKind.READ,
        resource_effect=KvResourceEffect.NONE,
    )
    second = KvTraceEvent(
        sequence=1,
        token_start=4,
        token_count=2,
        page_id=3,
        block_id=8,
        kind=KvEventKind.READ,
        resource_effect=KvResourceEffect.NONE,
    )
    trace = KvTrace((first, second))
    assert trace.events_for_token(3) == (first,)
    assert trace.events_for_token(4) == (second,)


def test_unexposed_bytes_never_become_zero() -> None:
    trace = KvTrace(
        (
            KvTraceEvent(
                sequence=0,
                token_start=0,
                token_count=1,
                page_id=0,
                block_id=0,
                kind=KvEventKind.READ,
                resource_effect=KvResourceEffect.TRANSFER,
                bytes_observed=32,
            ),
            KvTraceEvent(
                sequence=1,
                token_start=1,
                token_count=1,
                page_id=0,
                block_id=0,
                kind=KvEventKind.READ,
                resource_effect=KvResourceEffect.TRANSFER,
                bytes_observed=None,
            ),
        )
    )
    assert trace.bytes_by_effect(KvResourceEffect.TRANSFER) is None
    assert trace.bytes_by_effect(KvResourceEffect.LOGICAL_SIZE) == 0


def test_move_requires_distinct_explicit_tiers() -> None:
    with pytest.raises(ValueError, match="source_tier and destination_tier"):
        KvTraceEvent(
            sequence=0,
            token_start=0,
            token_count=1,
            page_id=0,
            block_id=0,
            kind=KvEventKind.MOVE,
            resource_effect=KvResourceEffect.TRANSFER,
        )
    with pytest.raises(ValueError, match="distinct"):
        KvTraceEvent(
            sequence=0,
            token_start=0,
            token_count=1,
            page_id=0,
            block_id=0,
            kind=KvEventKind.MOVE,
            resource_effect=KvResourceEffect.TRANSFER,
            source_tier="gpu",
            destination_tier="gpu",
        )


def test_paging_event_does_not_imply_logical_size_reduction() -> None:
    event = KvTraceEvent(
        sequence=0,
        token_start=0,
        token_count=16,
        page_id=4,
        block_id=1,
        kind=KvEventKind.ALLOCATE,
        resource_effect=KvResourceEffect.RESIDENCY,
        bytes_observed=4096,
    )
    trace = KvTrace((event,))
    assert trace.bytes_by_effect(KvResourceEffect.RESIDENCY) == 4096
    assert trace.bytes_by_effect(KvResourceEffect.LOGICAL_SIZE) == 0
