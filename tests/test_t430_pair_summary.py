import csv
import tempfile
import unittest
from pathlib import Path

from kvlab.t430_pair_summary import REQUIRED_FIELDS, load_samples, summarize


class T430PairSummaryTests(unittest.TestCase):
    def write_rows(self, rows):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "samples.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def row(repetition, *, mono, pair_wall, merge, selected=7, exact="exact"):
        odd = repetition % 2 == 1
        return {
            "repetition": str(repetition),
            "execution_order": "monolithic-first" if odd else "dual-first",
            "shard_launch_order": "shard0-first" if odd else "shard1-first",
            "monolithic_e2e_ns": str(mono),
            "shard0_scan_ns": "80",
            "shard1_scan_ns": "90",
            "pair_scan_wall_ns": str(pair_wall),
            "merge_ns": str(merge),
            "pair_e2e_ns": str(pair_wall + merge),
            "selected_pages": str(selected),
            "candidate_equality": exact,
        }

    def test_summary_uses_paired_end_to_end_medians(self):
        rows = load_samples(
            self.write_rows(
                [
                    self.row(1, mono=150, pair_wall=100, merge=10),
                    self.row(2, mono=180, pair_wall=110, merge=20),
                    self.row(3, mono=210, pair_wall=120, merge=30),
                ]
            )
        )
        result = summarize(rows, total_pages=320, signature_bits=256)
        self.assertEqual(result["monolithic_e2e_median_ns"], "180")
        self.assertEqual(result["dual_pair_e2e_median_ns"], "130")
        self.assertEqual(result["merge_median_ns"], "20")
        self.assertEqual(result["candidate_equality"], "exact-every-repetition")

    def test_rejects_independent_or_incomplete_pair_timing(self):
        row = self.row(1, mono=150, pair_wall=100, merge=10)
        row["pair_e2e_ns"] = "100"
        with self.assertRaisesRegex(ValueError, "must include"):
            load_samples(self.write_rows([row]))

    def test_rejects_non_exact_candidate_sample(self):
        rows = [self.row(1, mono=150, pair_wall=100, merge=10, exact="different")]
        with self.assertRaisesRegex(ValueError, "candidate equality"):
            load_samples(self.write_rows(rows))

    def test_rejects_non_alternated_multi_sample_campaign(self):
        rows = [
            self.row(1, mono=150, pair_wall=100, merge=10),
            self.row(2, mono=160, pair_wall=110, merge=10),
        ]
        rows[1]["execution_order"] = "monolithic-first"
        with self.assertRaisesRegex(ValueError, "execution order"):
            load_samples(self.write_rows(rows))

    def test_rejects_non_alternated_shard_launch_order(self):
        rows = [
            self.row(1, mono=150, pair_wall=100, merge=10),
            self.row(2, mono=160, pair_wall=110, merge=10),
            self.row(3, mono=170, pair_wall=120, merge=10),
        ]
        rows[2]["shard_launch_order"] = "shard1-first"
        with self.assertRaisesRegex(ValueError, "shard launch order"):
            load_samples(self.write_rows(rows))

    def test_rejects_fewer_than_three_paired_samples(self):
        rows = [
            self.row(1, mono=150, pair_wall=100, merge=10),
            self.row(2, mono=160, pair_wall=110, merge=10),
        ]
        with self.assertRaisesRegex(ValueError, "at least three"):
            load_samples(self.write_rows(rows))


if __name__ == "__main__":
    unittest.main()
