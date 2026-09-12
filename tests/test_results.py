import unittest

from kvlab.results import ExperimentResult, OutcomeClass, ResultRegistry


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def result(experiment_id: str, outcome: OutcomeClass, **kwargs: str) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        commit_sha=COMMIT,
        mechanism="paged_kv",
        outcome=outcome,
        decision_rule="candidate quality loss <= preregistered threshold",
        evidence_ref=f"artifacts/{experiment_id}.json",
        **kwargs,
    )


class ExperimentResultTests(unittest.TestCase):
    def test_negative_outcome_is_preserved_as_first_class_record(self) -> None:
        record = result("exp-negative", OutcomeClass.CRITERION_NOT_MET)
        registry = ResultRegistry()
        registry.append(record)
        self.assertEqual(
            registry.by_outcome(OutcomeClass.CRITERION_NOT_MET), (record,)
        )

    def test_inconclusive_outcome_is_not_promoted_to_success(self) -> None:
        record = result("exp-inconclusive", OutcomeClass.INCONCLUSIVE)
        registry = ResultRegistry()
        registry.append(record)
        self.assertEqual(registry.records(), (record,))
        self.assertEqual(registry.by_outcome(OutcomeClass.CRITERION_MET), ())

    def test_not_applicable_requires_reason(self) -> None:
        with self.assertRaises(ValueError):
            result("exp-na", OutcomeClass.NOT_APPLICABLE)
        record = result(
            "exp-na-valid",
            OutcomeClass.NOT_APPLICABLE,
            applicability_reason="model architecture does not expose compatible KV heads",
        )
        self.assertEqual(record.outcome, OutcomeClass.NOT_APPLICABLE)

    def test_reason_is_rejected_for_applicable_outcome(self) -> None:
        with self.assertRaises(ValueError):
            result(
                "exp-success",
                OutcomeClass.CRITERION_MET,
                applicability_reason="not valid here",
            )

    def test_duplicate_experiment_id_is_rejected(self) -> None:
        registry = ResultRegistry()
        record = result("exp-1", OutcomeClass.CRITERION_MET)
        registry.append(record)
        with self.assertRaises(ValueError):
            registry.append(record)

    def test_full_commit_sha_is_mandatory(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentResult(
                experiment_id="exp-short-sha",
                commit_sha="deadbeef",
                mechanism="quantization",
                outcome=OutcomeClass.CRITERION_NOT_MET,
                decision_rule="fixed preregistered rule",
                evidence_ref="artifacts/short.json",
            )


if __name__ == "__main__":
    unittest.main()
