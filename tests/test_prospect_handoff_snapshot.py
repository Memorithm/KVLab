"""Structural snapshot regressions shared with FLAT's MAA evidence review.

These tests exercise the real handoff constructor and Hamming validation. They
are not measurements of numerical attention, traffic, throughput or quality.
"""

from dataclasses import FrozenInstanceError
import unittest

from kvlab.prospect_handoff import (
    PROSPECT_BKV_HANDOFF_SCHEMA_V1,
    ProspectBkvHandoffError,
    ProspectBkvHandoffV1,
)


ZERO = "0000000000000000"
ONE = "0000000000000001"


class ProspectHandoffSnapshotTests(unittest.TestCase):
    def make_handoff(self, **changes):
        fields = {
            "schema": PROSPECT_BKV_HANDOFF_SCHEMA_V1,
            "signature_bits": 1,
            "generation": 7,
            "query_words": (ZERO,),
            "page_words": ((ZERO,), (ONE,)),
            "max_distance": 0,
            "admitted_pages": (0,),
        }
        fields.update(changes)
        return ProspectBkvHandoffV1(**fields)

    def test_all_mutable_aliases_are_detached_before_publication(self):
        query = [ZERO]
        page_zero = [ZERO]
        page_one = [ONE]
        pages = [page_zero, page_one]
        admitted = [0]
        handoff = self.make_handoff(
            query_words=query, page_words=pages, admitted_pages=admitted
        )
        before = handoff.canonical_json()
        before_hash = hash(handoff)

        query[0] = ONE
        page_zero[0] = ONE
        page_one.append(ZERO)
        pages.append([ZERO])
        admitted[:] = [1]

        self.assertEqual(handoff.query_words, (ZERO,))
        self.assertEqual(handoff.page_words, ((ZERO,), (ONE,)))
        self.assertEqual(handoff.admitted_pages, (0,))
        self.assertEqual(handoff.canonical_json(), before)
        self.assertEqual(hash(handoff), before_hash)
        self.assertEqual(ProspectBkvHandoffV1.from_canonical_json(before), handoff)

    def test_immutable_outer_tuple_does_not_hide_mutable_inner_pages(self):
        first = [ZERO]
        second = [ONE]
        handoff = self.make_handoff(page_words=(first, second))
        first.clear()
        second[0] = ZERO
        self.assertEqual(handoff.page_words, ((ZERO,), (ONE,)))
        self.assertEqual(handoff, self.make_handoff())

    def test_snapshotting_does_not_modify_caller_containers(self):
        query = [ZERO]
        pages = [[ZERO], [ONE]]
        admitted = [0]
        handoff = self.make_handoff(
            query_words=query, page_words=pages, admitted_pages=admitted
        )
        self.assertEqual(query, [ZERO])
        self.assertEqual(pages, [[ZERO], [ONE]])
        self.assertEqual(admitted, [0])
        self.assertIsInstance(query, list)
        self.assertIsInstance(pages[0], list)
        self.assertIsInstance(handoff.query_words, tuple)
        self.assertIsInstance(handoff.page_words[0], tuple)
        self.assertIsInstance(handoff.admitted_pages, tuple)

    def test_canonical_wire_bytes_are_unchanged_for_valid_inputs(self):
        tuple_handoff = self.make_handoff()
        list_handoff = self.make_handoff(
            query_words=[ZERO], page_words=[[ZERO], [ONE]], admitted_pages=[0]
        )
        self.assertEqual(tuple_handoff.canonical_json(), list_handoff.canonical_json())
        self.assertEqual(tuple_handoff, list_handoff)
        self.assertEqual(
            ProspectBkvHandoffV1.from_canonical_json(list_handoff.canonical_json()),
            tuple_handoff,
        )

    def test_invalid_sequence_containers_fail_with_a_domain_error(self):
        invalid = (
            {"query_words": ZERO},
            {"query_words": None},
            {"page_words": {ZERO}},
            {"page_words": [ZERO]},
            {"admitted_pages": {0}},
            {"admitted_pages": None},
        )
        for fields in invalid:
            with self.subTest(fields=fields):
                with self.assertRaises(ProspectBkvHandoffError):
                    self.make_handoff(**fields)

    def test_snapshot_does_not_approve_incorrect_candidate_membership(self):
        with self.assertRaises(ProspectBkvHandoffError):
            self.make_handoff(admitted_pages=[1])
        with self.assertRaises(ProspectBkvHandoffError):
            self.make_handoff(admitted_pages=[False])
        with self.assertRaises(ProspectBkvHandoffError):
            self.make_handoff(query_words=[ONE])

    def test_empty_admission_snapshot_remains_empty_when_input_page_changes(self):
        page = [ONE]
        admitted = []
        handoff = self.make_handoff(page_words=[page], admitted_pages=admitted)
        page[0] = ZERO
        admitted.append(0)
        self.assertEqual(handoff.page_words, ((ONE,),))
        self.assertEqual(handoff.admitted_pages, ())
        self.assertEqual(
            ProspectBkvHandoffV1.from_canonical_json(handoff.canonical_json()), handoff
        )

    def test_published_fields_and_nested_words_cannot_be_normally_mutated(self):
        handoff = self.make_handoff(page_words=[[ZERO], [ONE]])
        with self.assertRaises(FrozenInstanceError):
            handoff.generation = 8
        with self.assertRaises(TypeError):
            handoff.page_words[0][0] = ONE
        with self.assertRaises(TypeError):
            handoff.admitted_pages[0] = 1


if __name__ == "__main__":
    unittest.main()
