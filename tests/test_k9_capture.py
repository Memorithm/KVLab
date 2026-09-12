import unittest

from kvlab.k9_capture import (
    CaptureManifest,
    TensorCapture,
    assert_disjoint_capture_splits,
    digest_bytes,
)


class K9CaptureTests(unittest.TestCase):
    def capture(self, sample_id: str, *, kind: str = "key") -> TensorCapture:
        return TensorCapture(
            sample_id=sample_id,
            layer=3,
            kind=kind,
            shape=(2, 8, 64),
            dtype="float16",
            content_sha256=digest_bytes([sample_id.encode("utf-8"), kind.encode("utf-8")]),
            rope_removed=(kind == "key"),
        )

    def manifest(self, split: str, sample_ids: tuple[str, ...]) -> CaptureManifest:
        return CaptureManifest(
            experiment_id="k9-nvidia-reproduction-001",
            repository_commit="278373e6b83eb278034106eb2c62e4d45e0fc0a6",
            model_id="example/model",
            model_revision="revision-a",
            tokenizer_revision="tokenizer-a",
            runtime="fixture-runtime",
            split=split,
            captures=tuple(self.capture(sample_id) for sample_id in sample_ids),
        )

    def test_fingerprint_is_order_independent_for_capture_records(self) -> None:
        first = self.manifest("calibration", ("sample-a", "sample-b"))
        second = CaptureManifest(
            experiment_id=first.experiment_id,
            repository_commit=first.repository_commit,
            model_id=first.model_id,
            model_revision=first.model_revision,
            tokenizer_revision=first.tokenizer_revision,
            runtime=first.runtime,
            split=first.split,
            captures=tuple(reversed(first.captures)),
        )
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_holdout_overlap_fails_closed(self) -> None:
        calibration = self.manifest("calibration", ("sample-a", "sample-b"))
        final_holdout = self.manifest("final-holdout", ("sample-b", "sample-c"))
        with self.assertRaisesRegex(ValueError, "sample-b"):
            assert_disjoint_capture_splits(calibration, final_holdout)

    def test_disjoint_splits_are_accepted(self) -> None:
        calibration = self.manifest("calibration", ("sample-a", "sample-b"))
        final_holdout = self.manifest("final-holdout", ("sample-c", "sample-d"))
        assert_disjoint_capture_splits(calibration, final_holdout)

    def test_value_capture_rejects_rope_removed_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "only meaningful for key captures"):
            TensorCapture(
                sample_id="sample-a",
                layer=0,
                kind="value",
                shape=(1, 2, 3),
                dtype="float32",
                content_sha256=digest_bytes([b"value"]),
                rope_removed=True,
            )

    def test_duplicate_tensor_identity_is_rejected(self) -> None:
        capture = self.capture("sample-a")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            CaptureManifest(
                experiment_id="k9-test",
                repository_commit="deadbeef",
                model_id="example/model",
                model_revision="r1",
                tokenizer_revision="t1",
                runtime="fixture-runtime",
                split="calibration",
                captures=(capture, capture),
            )

    def test_digest_requires_bytes_and_non_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            digest_bytes([])
        with self.assertRaisesRegex(TypeError, "must be bytes"):
            digest_bytes([b"ok", "not-bytes"])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
