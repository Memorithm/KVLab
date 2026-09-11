import unittest

from kvlab.oracle import KVLayerRecord, OracleError, capture_full_cache


class FullCacheOracleTests(unittest.TestCase):
    def _records(self):
        return (
            KVLayerRecord(0, "fp16", (1, 2, 3), (1, 2, 3), b"key0", b"val0"),
            KVLayerRecord(1, "fp16", (1, 2, 3), (1, 2, 3), b"key1", b"val1"),
        )

    def test_capture_is_deterministic_and_replay_is_exact(self):
        first = capture_full_cache(
            model_revision="model@abc",
            tokenizer_revision="tok@def",
            sequence_length=128,
            records=self._records(),
        )
        second = capture_full_cache(
            model_revision="model@abc",
            tokenizer_revision="tok@def",
            sequence_length=128,
            records=self._records(),
        )
        self.assertEqual(first.digest_sha256, second.digest_sha256)
        self.assertEqual(tuple(first.replay()), self._records())
        self.assertEqual(first.logical_bytes, 16)

    def test_payload_change_changes_digest(self):
        baseline = capture_full_cache(
            model_revision="model@abc",
            tokenizer_revision="tok@def",
            sequence_length=128,
            records=self._records(),
        )
        changed = list(self._records())
        changed[1] = KVLayerRecord(1, "fp16", (1, 2, 3), (1, 2, 3), b"key1", b"DIFF")
        candidate = capture_full_cache(
            model_revision="model@abc",
            tokenizer_revision="tok@def",
            sequence_length=128,
            records=changed,
        )
        self.assertNotEqual(baseline.digest_sha256, candidate.digest_sha256)

    def test_rejects_duplicate_or_unsorted_layers(self):
        duplicate = (self._records()[0], self._records()[0])
        with self.assertRaises(OracleError):
            capture_full_cache(
                model_revision="model@abc",
                tokenizer_revision="tok@def",
                sequence_length=128,
                records=duplicate,
            )
        with self.assertRaises(OracleError):
            capture_full_cache(
                model_revision="model@abc",
                tokenizer_revision="tok@def",
                sequence_length=128,
                records=tuple(reversed(self._records())),
            )

    def test_rejects_ambiguous_metadata(self):
        with self.assertRaises(OracleError):
            capture_full_cache(
                model_revision="",
                tokenizer_revision="tok@def",
                sequence_length=128,
                records=self._records(),
            )
        bad = KVLayerRecord(0, "fp 16", (1, 0), (1, 2), b"k", b"v")
        with self.assertRaises(OracleError):
            capture_full_cache(
                model_revision="model@abc",
                tokenizer_revision="tok@def",
                sequence_length=128,
                records=(bad,),
            )


if __name__ == "__main__":
    unittest.main()
