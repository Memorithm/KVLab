import hashlib
import json
import unittest

from kvlab.flat_bikv_selection_quality import (
    FLAT_BIKV_SELECTION_QUALITY_CANDIDATE_REVISION,
    FlatBikvSelectionQualityError,
    FlatBikvSelectionQualityV1,
)


def _compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _fnv(payload):
    value = 0xCBF29CE484222325
    for byte in payload:
        value ^= byte
        value = (value * 0x100000001B3) & ((1 << 64) - 1)
    return f"{value:016x}"


def _with_checksum(value):
    prefix = _compact(value)[:-1]
    result = dict(value)
    result["evidence_checksum"] = {
        "algorithm": "fnv1a64",
        "value": _fnv(prefix),
    }
    return result


def _selection(**updates):
    value = {
        "schema": "flat.boolean-kv-selection.v2",
        "generation": 7,
        "signature_bits": 64,
        "max_distance": 8,
        "live_tokens": 6,
        "mapped_pages": 3,
        "boolean_pages_scanned": 3,
        "boolean_key_bytes_read": 24,
        "numerical_kv_bytes_per_token": 8,
        "full_numerical_kv_bytes": 48,
        "selected_numerical_kv_bytes": 32,
        "avoided_numerical_kv_bytes": 16,
        "selected_pages": [
            {
                "logical_page": 0,
                "physical_page": 10,
                "live_tokens": 2,
                "hamming_distance": 1,
                "xnor_matches": 63,
            },
            {
                "logical_page": 2,
                "physical_page": 12,
                "live_tokens": 2,
                "hamming_distance": 2,
                "xnor_matches": 62,
            },
        ],
    }
    value.update(updates)
    return _with_checksum(value)


def _quality(selection=None, target=None, metric_updates=None):
    selection = selection or _selection()
    target = [0, 1] if target is None else target
    selected = tuple(page["logical_page"] for page in selection["selected_pages"])
    target_tuple = tuple(target)
    true_positive = len(set(selected).intersection(target_tuple))
    metrics = {
        "mapped_pages": selection["mapped_pages"],
        "selected_pages": len(selected),
        "target_pages": len(target_tuple),
        "true_positive_pages": true_positive,
        "false_negative_pages": len(target_tuple) - true_positive,
        "false_positive_pages": len(selected) - true_positive,
        "recall": {"numerator": true_positive, "denominator": len(target_tuple)},
        "false_negative_rate": {
            "numerator": len(target_tuple) - true_positive,
            "denominator": len(target_tuple),
        },
        "candidate_density": {
            "numerator": len(selected),
            "denominator": selection["mapped_pages"],
        },
    }
    if metric_updates:
        metrics.update(metric_updates)
    core = {
        "schema": "flat.bikv-selection-quality.v1",
        "selection": selection,
        "declared_dense_target_pages": target,
        "metrics": metrics,
    }
    prefix = _compact(core)[:-1]
    core["quality_checksum"] = {
        "algorithm": "fnv1a64",
        "value": _fnv(prefix),
    }
    return _compact(core)


class FlatBikvSelectionQualityTests(unittest.TestCase):
    def test_accepts_canonical_quality_and_recomputes_exact_metrics(self):
        payload = _quality()
        evidence = FlatBikvSelectionQualityV1.from_canonical_json_bytes(payload)
        self.assertEqual(evidence.canonical_bytes, payload)
        self.assertEqual(evidence.target_pages, (0, 1))
        self.assertEqual(evidence.true_positive_pages, 1)
        self.assertEqual(evidence.false_negative_pages, 1)
        self.assertEqual(evidence.false_positive_pages, 1)
        self.assertEqual(evidence.quality_sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(len(evidence.selection_sha256), 64)
        self.assertEqual(
            FLAT_BIKV_SELECTION_QUALITY_CANDIDATE_REVISION,
            "6fdcaf38c1be7d7f4a70fac6d7b8aa6f214db124",
        )

    def test_rejects_metric_drift(self):
        payload = _quality(metric_updates={"false_negative_pages": 0})
        with self.assertRaisesRegex(
            FlatBikvSelectionQualityError, "false_negative_pages"
        ):
            FlatBikvSelectionQualityV1.from_canonical_json_bytes(payload)

    def test_rejects_fraction_drift(self):
        payload = _quality(
            metric_updates={"recall": {"numerator": 2, "denominator": 2}}
        )
        with self.assertRaisesRegex(FlatBikvSelectionQualityError, "metrics.recall"):
            FlatBikvSelectionQualityV1.from_canonical_json_bytes(payload)

    def test_rejects_stale_outer_checksum_after_target_or_metric_mutation(self):
        payload = _quality()
        raw = json.loads(payload)
        raw["declared_dense_target_pages"] = [0]
        mutated_target = _compact(raw)
        with self.assertRaisesRegex(FlatBikvSelectionQualityError, "quality_checksum"):
            FlatBikvSelectionQualityV1.from_canonical_json_bytes(mutated_target)

        raw = json.loads(payload)
        raw["metrics"]["false_positive_pages"] = 0
        mutated_metric = _compact(raw)
        with self.assertRaisesRegex(FlatBikvSelectionQualityError, "false_positive_pages|quality_checksum"):
            FlatBikvSelectionQualityV1.from_canonical_json_bytes(mutated_metric)

    def test_rejects_empty_unsorted_duplicate_and_out_of_range_targets(self):
        cases = (
            ([], "non-empty"),
            ([1, 0], "strictly increasing"),
            ([1, 1], "strictly increasing"),
            ([3], "outside 3 mapped pages"),
        )
        for target, message in cases:
            with self.subTest(target=target):
                payload = _quality(target=target)
                with self.assertRaisesRegex(FlatBikvSelectionQualityError, message):
                    FlatBikvSelectionQualityV1.from_canonical_json_bytes(payload)

    def test_rejects_boolean_integer_and_invalid_embedded_selection(self):
        payload = _quality(metric_updates={"mapped_pages": True})
        with self.assertRaisesRegex(FlatBikvSelectionQualityError, "integer u64"):
            FlatBikvSelectionQualityV1.from_canonical_json_bytes(payload)

        bad_selection = _selection(max_distance=65)
        payload = _quality(selection=bad_selection)
        with self.assertRaisesRegex(FlatBikvSelectionQualityError, "embedded selection"):
            FlatBikvSelectionQualityV1.from_canonical_json_bytes(payload)

    def test_rejects_duplicate_keys_and_noncanonical_bytes(self):
        payload = _quality()
        duplicated = payload.replace(
            b'{"schema":"flat.bikv-selection-quality.v1",',
            b'{"schema":"flat.bikv-selection-quality.v1","schema":"flat.bikv-selection-quality.v1",',
            1,
        )
        with self.assertRaisesRegex(FlatBikvSelectionQualityError, "duplicate JSON key"):
            FlatBikvSelectionQualityV1.from_canonical_json_bytes(duplicated)
        with self.assertRaisesRegex(FlatBikvSelectionQualityError, "canonical"):
            FlatBikvSelectionQualityV1.from_canonical_json_bytes(payload + b"\n")


if __name__ == "__main__":
    unittest.main()
