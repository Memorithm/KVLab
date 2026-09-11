import unittest

from kvlab.strategy import (
    KvStrategy,
    StrategyAction,
    StrategyContext,
    StrategyDecision,
    validate_decisions,
)


class KeepAllStrategy:
    @property
    def name(self) -> str:
        return "keep-all"

    def decide(self, context: StrategyContext) -> tuple[StrategyDecision, ...]:
        if context.token_count == 0:
            return ()
        return (
            StrategyDecision(
                action=StrategyAction.KEEP,
                token_start=0,
                token_count=context.token_count,
            ),
        )


class StrategyContractTests(unittest.TestCase):
    def test_structural_protocol_accepts_strategy(self) -> None:
        strategy = KeepAllStrategy()
        self.assertIsInstance(strategy, KvStrategy)
        context = StrategyContext(token_count=4, logical_kv_bytes=1024)
        decisions = strategy.decide(context)
        validate_decisions(context, decisions)
        self.assertEqual(decisions[0].action, StrategyAction.KEEP)

    def test_unexposed_capacity_remains_none(self) -> None:
        context = StrategyContext(token_count=2, logical_kv_bytes=512)
        self.assertIsNone(context.available_gpu_bytes)
        self.assertIsNone(context.available_host_bytes)

    def test_move_requires_explicit_target_tier(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_tier"):
            StrategyDecision(
                action=StrategyAction.MOVE,
                token_start=0,
                token_count=1,
            )

    def test_quantize_requires_explicit_representation(self) -> None:
        with self.assertRaisesRegex(ValueError, "representation"):
            StrategyDecision(
                action=StrategyAction.QUANTIZE,
                token_start=0,
                token_count=1,
            )

    def test_decision_outside_observed_range_fails_closed(self) -> None:
        context = StrategyContext(token_count=4, logical_kv_bytes=1024)
        decisions = (
            StrategyDecision(
                action=StrategyAction.EVICT,
                token_start=3,
                token_count=2,
            ),
        )
        with self.assertRaisesRegex(ValueError, "exceeds observed token range"):
            validate_decisions(context, decisions)


if __name__ == "__main__":
    unittest.main()
