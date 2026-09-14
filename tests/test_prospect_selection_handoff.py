import json

import pytest

from kvlab.prospect_selection_handoff import (
    PROSPECT_KV_SELECTION_HANDOFF_SCHEMA_V1,
    ProspectKvSelectionHandoffError,
    ProspectKvSelectionHandoffV1,
)


def test_capture_round_trips_arbitrary_explicit_selection() -> None:
    handoff = ProspectKvSelectionHandoffV1.capture(
        token_ids=(10, 11, 12, 13, 14),
        bytes_per_token=64,
        policy="lru",
        retained_token_ids=(10, 12, 14),
    )

    assert handoff.schema == PROSPECT_KV_SELECTION_HANDOFF_SCHEMA_V1
    assert handoff.policy == "lru"
    assert handoff.retained_token_ids == (10, 12, 14)
    assert handoff.evicted_token_ids == (11, 13)
    assert handoff.logical_input_bytes == 320
    assert handoff.logical_retained_bytes == 192
    assert handoff.logical_evicted_bytes == 128

    payload = handoff.canonical_json()
    assert ProspectKvSelectionHandoffV1.from_canonical_json(payload) == handoff


def test_capture_rejects_unknown_or_reordered_retained_tokens() -> None:
    with pytest.raises(ProspectKvSelectionHandoffError):
        ProspectKvSelectionHandoffV1.capture(
            token_ids=(10, 11, 12),
            bytes_per_token=64,
            policy="magnitude",
            retained_token_ids=(10, 99),
        )

    with pytest.raises(ProspectKvSelectionHandoffError):
        ProspectKvSelectionHandoffV1.capture(
            token_ids=(10, 11, 12),
            bytes_per_token=64,
            policy="magnitude",
            retained_token_ids=(12, 10),
        )


def test_decoder_rejects_partition_or_byte_tampering() -> None:
    handoff = ProspectKvSelectionHandoffV1.capture(
        token_ids=(10, 11, 12, 13),
        bytes_per_token=32,
        policy="sensitivity_per_byte",
        retained_token_ids=(11, 13),
    )
    raw = json.loads(handoff.canonical_json())
    raw["evicted_token_ids"] = [10]
    tampered = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ProspectKvSelectionHandoffError):
        ProspectKvSelectionHandoffV1.from_canonical_json(tampered)

    raw = json.loads(handoff.canonical_json())
    raw["logical_retained_bytes"] = 1
    tampered = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ProspectKvSelectionHandoffError):
        ProspectKvSelectionHandoffV1.from_canonical_json(tampered)


def test_decoder_rejects_noncanonical_json_and_unknown_fields() -> None:
    handoff = ProspectKvSelectionHandoffV1.capture(
        token_ids=(1, 2),
        bytes_per_token=16,
        policy="random_seed_7",
        retained_token_ids=(2,),
    )
    raw = json.loads(handoff.canonical_json())

    noncanonical = json.dumps(raw, sort_keys=True, indent=2)
    with pytest.raises(ProspectKvSelectionHandoffError):
        ProspectKvSelectionHandoffV1.from_canonical_json(noncanonical)

    raw["unexpected"] = 1
    unknown = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ProspectKvSelectionHandoffError):
        ProspectKvSelectionHandoffV1.from_canonical_json(unknown)


def test_full_eviction_is_explicitly_representable() -> None:
    handoff = ProspectKvSelectionHandoffV1.capture(
        token_ids=(4, 5),
        bytes_per_token=8,
        policy="explicit-empty-control",
        retained_token_ids=(),
    )
    assert handoff.retained_token_ids == ()
    assert handoff.evicted_token_ids == (4, 5)
    assert handoff.logical_retained_bytes == 0
    assert handoff.logical_evicted_bytes == 16
