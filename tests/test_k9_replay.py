import unittest

from kvlab.k9_capture import CaptureManifest, TensorCapture, digest_bytes
from kvlab.k9_replay import CaptureIdentity, ReplayPayload, validate_replay_payloads


class K9ReplayTests(unittest.TestCase):
    def manifest(self) -> CaptureManifest:
        key_bytes = b"key-payload"
        value_bytes = b"value-payload"
        return CaptureManifest(
            experiment_id="k9-replay-fixture",
            repository_commit="7127ac33d710703a6e0f0421d262d158e7eba074",
            model_id="fixture/model",
            model_revision="revision-a",
            tokenizer_revision="tokenizer-a",
            runtime="fixture-runtime",
            split="calibration",
            captures=(
                TensorCapture(
                    sample_id="sample-a",
                    layer=2,
                    kind="key",
                    shape=(1, 2, 64),
                    dtype="float16",
                    content_sha256=digest_bytes([key_bytes]),
                    rope_removed=True,
                ),
                TensorCapture(
                    sample_id="sample-a",
                    layer=2,
                    kind="value",
                    shape=(1, 2, 64),
                    dtype="float16",
                    content_sha256=digest_bytes([value_bytes]),
                    rope_removed=False,
                ),
            ),
        )

    def payload(self, kind: str, data: bytes) -> ReplayPayload:
        return ReplayPayload(
            identity=CaptureIdentity("sample-a", 2, kind),
            chunks=(data[:3], data[3:]),
        )

    def test_exact_payloads_validate_and_report_observed_bytes(self) -> None:
        lengths = validate_replay_payloads(
            self.manifest(),
            (
                self.payload("value", b"value-payload"),
                self.payload("key", b"key-payload"),
            ),
        )
        self.assertEqual(lengths[CaptureIdentity("sample-a", 2, "key")], 11)
        self.assertEqual(lengths[CaptureIdentity("sample-a", 2, "value")], 13)

    def test_corrupted_payload_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            validate_replay_payloads(
                self.manifest(),
                (
                    self.payload("key", b"corrupted"),
                    self.payload("value", b"value-payload"),
                ),
            )

    def test_missing_payload_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing replay payloads"):
            validate_replay_payloads(
                self.manifest(),
                (self.payload("key", b"key-payload"),),
            )

    def test_unexpected_payload_fails_closed(self) -> None:
        extra = ReplayPayload(
            identity=CaptureIdentity("sample-b", 2, "key"),
            chunks=(b"extra",),
        )
        with self.assertRaisesRegex(ValueError, "unexpected replay payloads"):
            validate_replay_payloads(
                self.manifest(),
                (
                    self.payload("key", b"key-payload"),
                    self.payload("value", b"value-payload"),
                    extra,
                ),
            )

    def test_duplicate_payload_fails_closed(self) -> None:
        key = self.payload("key", b"key-payload")
        with self.assertRaisesRegex(ValueError, "duplicate replay payload"):
            validate_replay_payloads(
                self.manifest(),
                (key, key, self.payload("value", b"value-payload")),
            )

    def test_payload_chunks_must_be_bytes_and_non_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            ReplayPayload(CaptureIdentity("sample-a", 0, "key"), ())
        with self.assertRaisesRegex(TypeError, "must be bytes"):
            ReplayPayload(  # type: ignore[arg-type]
                CaptureIdentity("sample-a", 0, "key"),
                (b"ok", "bad"),
            )


if __name__ == "__main__":
    unittest.main()
