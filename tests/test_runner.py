import json
import unittest

from kvlab.instrumentation import KVResourceAccounting, MeasurementKind, bytes_quantity, not_exposed as resource_not_exposed, tokens_quantity
from kvlab.latency import KVLatencyAccounting, measured, not_exposed as latency_not_exposed
from kvlab.oracle import KVLayerRecord, capture_full_cache
from kvlab.runner import ExperimentRunRecord, RunRecordError
from kvlab.trace import KvTrace, KvTraceEvent, KvEventKind, KvResourceEffect


class ExperimentRunRecordTests(unittest.TestCase):
    def oracle(self):
        return capture_full_cache(
            model_revision="model-rev",
            tokenizer_revision="tok-rev",
            sequence_length=1,
            records=(
                KVLayerRecord(
                    layer=0,
                    dtype="fp16",
                    key_shape=(1, 1, 1, 2),
                    value_shape=(1, 1, 1, 2),
                    key_bytes=b"abcd",
                    value_bytes=b"efgh",
                ),
            ),
        )

    def resources(self, logical_bytes=8):
        return KVResourceAccounting(
            logical_cache_bytes=bytes_quantity(logical_bytes, MeasurementKind.MEASURED),
            gpu_resident_bytes=resource_not_exposed("bytes"),
            host_resident_bytes=resource_not_exposed("bytes"),
            secondary_storage_bytes=resource_not_exposed("bytes"),
            fragmentation_bytes=resource_not_exposed("bytes"),
            host_to_gpu_bytes=resource_not_exposed("bytes"),
            gpu_to_host_bytes=resource_not_exposed("bytes"),
            bytes_read=resource_not_exposed("bytes"),
            bytes_written=resource_not_exposed("bytes"),
            recomputed_token_count=tokens_quantity(0, MeasurementKind.MEASURED),
        )

    def latency(self):
        return KVLatencyAccounting(
            prefill=measured(1.25),
            ttft=latency_not_exposed(),
            decode_total=latency_not_exposed(),
            tpot=latency_not_exposed(),
            cache_transform=latency_not_exposed(),
            cache_reconstruction=latency_not_exposed(),
        )

    def trace(self):
        return KvTrace(
            events=(
                KvTraceEvent(
                    sequence=0,
                    token_start=0,
                    token_count=1,
                    page_id=None,
                    block_id=None,
                    kind=KvEventKind.ALLOCATE,
                    resource_effect=KvResourceEffect.LOGICAL_SIZE,
                    bytes_observed=8,
                ),
            )
        )

    def test_record_binds_oracle_and_preserves_unexposed_metrics(self):
        oracle = self.oracle()
        record = ExperimentRunRecord.from_observations(
            experiment_id="k1-smoke-001",
            repository_revision="a" * 40,
            oracle=oracle,
            resources=self.resources(),
            latency=self.latency(),
            trace=self.trace(),
        )
        payload = json.loads(record.to_json())
        self.assertEqual(payload["oracle_digest_sha256"], oracle.digest_sha256)
        self.assertEqual(payload["oracle_logical_bytes"], 8)
        self.assertIsNone(payload["resources"]["gpu_resident_bytes"]["value"])
        self.assertEqual(payload["resources"]["gpu_resident_bytes"]["kind"], "not_exposed")

    def test_logical_size_must_match_full_cache_oracle(self):
        with self.assertRaisesRegex(RunRecordError, "logical-cache"):
            ExperimentRunRecord.from_observations(
                experiment_id="k1-smoke-001",
                repository_revision="b" * 40,
                oracle=self.oracle(),
                resources=self.resources(logical_bytes=9),
                latency=self.latency(),
                trace=self.trace(),
            )

    def test_repository_revision_requires_full_lowercase_sha(self):
        with self.assertRaisesRegex(RunRecordError, "full Git SHA"):
            ExperimentRunRecord.from_observations(
                experiment_id="k1-smoke-001",
                repository_revision="deadbeef",
                oracle=self.oracle(),
                resources=self.resources(),
                latency=self.latency(),
                trace=self.trace(),
            )


if __name__ == "__main__":
    unittest.main()
