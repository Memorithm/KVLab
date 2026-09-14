import unittest
from dataclasses import replace

from kvlab.oracle import KVLayerRecord, OracleError, capture_full_cache


class FullCacheIntegrityTests(unittest.TestCase):
    def records(self):
        return (
            KVLayerRecord(0, "fp16", (1, 2, 3), (1, 2, 3), b"key0", b"val0"),
            KVLayerRecord(1, "fp16", (1, 2, 3), (1, 2, 3), b"key1", b"val1"),
        )

    def snapshot(self, **kwargs):
        args = dict(model_revision="model@abc", tokenizer_revision="tok@def",
                    sequence_length=128, records=self.records())
        args.update(kwargs)
        return capture_full_cache(**args)

    def test_v1_digest_and_record_bytes_remain_compatible(self):
        snapshot = self.snapshot()
        self.assertEqual(snapshot.digest_sha256,
                         "f1e6794bed8027b20c9779ed66508aca98da3325047a620f0b799a26c7b2fd73")
        self.assertEqual(tuple(snapshot.replay()), self.records())
        self.assertEqual(snapshot.logical_bytes, 16)

    def test_unknown_or_mistyped_schema_fails_before_first_yield(self):
        snapshot = self.snapshot()
        for version in (0, 2, 999, True, False, 1.0, "1", None):
            with self.subTest(version=version):
                replay = replace(snapshot, schema_version=version).replay()
                with self.assertRaisesRegex(OracleError, "schema version"):
                    next(replay)

    def test_capture_freezes_caller_owned_nested_shapes(self):
        key_shape, value_shape = [1, 2, 3], [1, 2, 3]
        records = [replace(self.records()[0], key_shape=key_shape, value_shape=value_shape)]
        snapshot = self.snapshot(records=records)
        digest = snapshot.digest_sha256
        key_shape[0] = 99
        value_shape.clear()
        records.clear()
        self.assertEqual(snapshot.digest_sha256, digest)
        self.assertEqual(tuple(snapshot.replay()), (self.records()[0],))
        self.assertIsInstance(snapshot.records[0].key_shape, tuple)

    def test_direct_snapshot_replay_uses_verified_record_list(self):
        records = list(self.records())
        snapshot = replace(self.snapshot(), records=records)
        replay = snapshot.replay()
        self.assertEqual(next(replay), self.records()[0])
        records[1] = replace(records[1], key_bytes=b"CHANGED")
        self.assertEqual(next(replay), self.records()[1])
        with self.assertRaises(StopIteration):
            next(replay)
        # A newly started replay still detects the tampered original container.
        with self.assertRaisesRegex(OracleError, "digest mismatch"):
            next(snapshot.replay())

    def test_direct_snapshot_replay_freezes_shapes_before_yield(self):
        shape = [1, 2, 3]
        records = [self.records()[0], replace(self.records()[1], key_shape=shape)]
        snapshot = replace(self.snapshot(), records=records)
        replay = snapshot.replay()
        next(replay)
        shape[0] = 99
        second = next(replay)
        self.assertEqual(second.key_shape, (1, 2, 3))
        self.assertIsInstance(second.key_shape, tuple)
        with self.assertRaisesRegex(OracleError, "digest mismatch"):
            next(snapshot.replay())

    def test_integer_metadata_rejects_boolean_and_non_integer_values(self):
        for value in (True, 1.0, "1", None, -1):
            with self.subTest(value=value):
                with self.assertRaises(OracleError):
                    self.snapshot(sequence_length=value)
                for field in ("layer", "key_shape", "value_shape"):
                    record = replace(self.records()[0], **{
                        field: value if field == "layer" else (value,)
                    })
                    with self.assertRaises(OracleError):
                        self.snapshot(records=(record,))

    def test_invalid_header_or_records_reject_before_any_yield(self):
        snapshot = self.snapshot()
        for field in ("model_revision", "tokenizer_revision"):
            for value in (None, 12, "", " "):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(OracleError):
                        self.snapshot(**{field: value})
                    with self.assertRaises(OracleError):
                        next(replace(snapshot, **{field: value}).replay())
        for records in (None, (), ("not-a-record",), tuple(reversed(self.records()))):
            with self.subTest(records=records):
                with self.assertRaises(OracleError):
                    self.snapshot(records=records)
                with self.assertRaises(OracleError):
                    next(replace(snapshot, records=records).replay())

    def test_dtype_and_payload_semantics_remain_explicit_not_inferred(self):
        opaque = KVLayerRecord(0, "custom-codec", (3, 7), (2, 9), b"k", b"v")
        snapshot = self.snapshot(records=(opaque,))
        self.assertEqual(tuple(snapshot.replay()), (opaque,))
        for record in (
            replace(opaque, dtype=42),
            replace(opaque, key_shape="37"),
            replace(opaque, value_shape={2, 9}),
            replace(opaque, key_bytes=bytearray(b"k")),
        ):
            with self.subTest(record=record):
                with self.assertRaises(OracleError):
                    self.snapshot(records=(record,))


if __name__ == "__main__":
    unittest.main()
