import unittest

from kvlab.tier_events import (
    TierMovementEvent,
    TierMovementKind,
    summarize_tier_movements,
)
from kvlab.tiering import Tier


class TierMovementEventTests(unittest.TestCase):
    def test_explicit_events_account_directional_bytes(self) -> None:
        summary = summarize_tier_movements(
            (
                TierMovementEvent(
                    page_id=1,
                    source=Tier.SECONDARY,
                    destination=Tier.HOST,
                    bytes_moved=4096,
                    kind=TierMovementKind.PROMOTE,
                ),
                TierMovementEvent(
                    page_id=2,
                    source=Tier.HOST,
                    destination=Tier.GPU,
                    bytes_moved=4096,
                    kind=TierMovementKind.PROMOTE,
                ),
                TierMovementEvent(
                    page_id=3,
                    source=Tier.GPU,
                    destination=Tier.HOST,
                    bytes_moved=2048,
                    kind=TierMovementKind.DEMOTE,
                ),
            )
        )

        self.assertEqual(summary.event_count, 3)
        self.assertEqual(summary.promoted_bytes, 8192)
        self.assertEqual(summary.demoted_bytes, 2048)
        self.assertEqual(summary.total_measured_transfer_bytes, 10240)
        self.assertEqual(summary.gpu_in_bytes, 4096)
        self.assertEqual(summary.gpu_out_bytes, 2048)
        self.assertEqual(summary.host_in_bytes, 6144)
        self.assertEqual(summary.host_out_bytes, 4096)
        self.assertEqual(summary.secondary_in_bytes, 0)
        self.assertEqual(summary.secondary_out_bytes, 4096)

    def test_empty_event_stream_claims_zero_measured_traffic_only(self) -> None:
        summary = summarize_tier_movements(())
        self.assertEqual(summary.event_count, 0)
        self.assertEqual(summary.total_measured_transfer_bytes, 0)

    def test_invalid_events_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            TierMovementEvent(
                page_id=-1,
                source=Tier.HOST,
                destination=Tier.GPU,
                bytes_moved=4096,
                kind=TierMovementKind.PROMOTE,
            )
        with self.assertRaises(ValueError):
            TierMovementEvent(
                page_id=0,
                source=Tier.GPU,
                destination=Tier.GPU,
                bytes_moved=4096,
                kind=TierMovementKind.PROMOTE,
            )
        with self.assertRaises(ValueError):
            TierMovementEvent(
                page_id=0,
                source=Tier.GPU,
                destination=Tier.HOST,
                bytes_moved=0,
                kind=TierMovementKind.DEMOTE,
            )


if __name__ == "__main__":
    unittest.main()
