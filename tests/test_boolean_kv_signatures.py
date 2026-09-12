import unittest

from kvlab.boolean_kv import pack_bits
from kvlab.boolean_kv_signatures import (
    SignatureControlError,
    aggregate_page_signatures,
    evaluate_candidate_selection,
    positional_control_signature,
    random_control_signature,
    sign_projection,
)


class SignatureControlTests(unittest.TestCase):
    def test_random_control_is_reproducible_and_identity_bound(self) -> None:
        left = random_control_signature(bit_length=130, seed=7, identity=3)
        right = random_control_signature(bit_length=130, seed=7, identity=3)
        other = random_control_signature(bit_length=130, seed=7, identity=4)
        self.assertEqual(left, right)
        self.assertNotEqual(left, other)
        self.assertEqual(left.bit_length, 130)

    def test_sign_projection_has_explicit_threshold_semantics(self) -> None:
        signature = sign_projection([-1.0, 0.0, 0.25, 1.0], threshold=0.25)
        self.assertEqual(signature, pack_bits([False, False, True, True]))
        with self.assertRaises(SignatureControlError):
            sign_projection([0.0, float("nan")])

    def test_positional_control_depends_only_on_frozen_structure(self) -> None:
        first = positional_control_signature(
            bit_length=128,
            logical_page=2,
            token_start=64,
            token_count=32,
        )
        again = positional_control_signature(
            bit_length=128,
            logical_page=2,
            token_start=64,
            token_count=32,
        )
        moved = positional_control_signature(
            bit_length=128,
            logical_page=3,
            token_start=96,
            token_count=32,
        )
        self.assertEqual(first, again)
        self.assertNotEqual(first, moved)

    def test_page_aggregation_or_and_majority_are_frozen(self) -> None:
        signatures = [
            pack_bits([True, False, True, False]),
            pack_bits([True, True, False, False]),
            pack_bits([False, True, True, False]),
        ]
        self.assertEqual(
            aggregate_page_signatures(signatures, policy="or"),
            pack_bits([True, True, True, False]),
        )
        self.assertEqual(
            aggregate_page_signatures(signatures, policy="and"),
            pack_bits([False, False, False, False]),
        )
        self.assertEqual(
            aggregate_page_signatures(signatures, policy="majority"),
            pack_bits([True, True, True, False]),
        )

    def test_majority_tie_is_false(self) -> None:
        signatures = [pack_bits([True]), pack_bits([False])]
        self.assertEqual(
            aggregate_page_signatures(signatures, policy="majority"),
            pack_bits([False]),
        )

    def test_selection_metrics_report_recall_false_negatives_and_density(self) -> None:
        metrics = evaluate_candidate_selection(
            selected_pages=[1, 2, 5],
            dense_target_pages=[1, 2, 3, 4],
            total_pages=8,
        )
        self.assertEqual(metrics.true_positives, 2)
        self.assertEqual(metrics.false_negatives, 2)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.false_negative_rate, 0.5)
        self.assertEqual(metrics.candidate_density, 3 / 8)

    def test_selection_validation_fails_closed(self) -> None:
        with self.assertRaises(SignatureControlError):
            evaluate_candidate_selection(
                selected_pages=[], dense_target_pages=[], total_pages=4
            )
        with self.assertRaises(SignatureControlError):
            evaluate_candidate_selection(
                selected_pages=[4], dense_target_pages=[0], total_pages=4
            )
        with self.assertRaises(SignatureControlError):
            aggregate_page_signatures([pack_bits([True])], policy="unknown")


if __name__ == "__main__":
    unittest.main()
