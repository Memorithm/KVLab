import unittest

from kvlab.tier_events import TierMovementEvent, TierMovementKind
from kvlab.tiering import Tier


class TierMovementDirectionTests(unittest.TestCase):
    def test_promotion_must_move_toward_gpu(self) -> None:
        TierMovementEvent(
            page_id=0,
            source=Tier.SECONDARY,
            destination=Tier.GPU,
            bytes_moved=4096,
            kind=TierMovementKind.PROMOTE,
        )

        with self.assertRaises(ValueError):
            TierMovementEvent(
                page_id=1,
                source=Tier.GPU,
                destination=Tier.HOST,
                bytes_moved=4096,
                kind=TierMovementKind.PROMOTE,
            )

    def test_demotion_must_move_away_from_gpu(self) -> None:
        TierMovementEvent(
            page_id=0,
            source=Tier.GPU,
            destination=Tier.SECONDARY,
            bytes_moved=4096,
            kind=TierMovementKind.DEMOTE,
        )

        with self.assertRaises(ValueError):
            TierMovementEvent(
                page_id=1,
                source=Tier.SECONDARY,
                destination=Tier.HOST,
                bytes_moved=4096,
                kind=TierMovementKind.DEMOTE,
            )


if __name__ == "__main__":
    unittest.main()
