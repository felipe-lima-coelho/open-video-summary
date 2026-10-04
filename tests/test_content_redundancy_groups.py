"""Selected redundancy pairs form disjoint connected components."""

import itertools
import unittest
from dataclasses import asdict

from open_video_summary.core.selection_criteria.redundancy import ContentBasedRedundancy
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.handlers.summary import SummarySegmentHandler


class ContentRedundancyGroupingTests(unittest.TestCase):
    def setUp(self):
        self.videos = [
            Video(
                label, f"data/raw/{label}.mp4",
                segments=[VideoSegment(f"Transcript {label}", 0, 5, order=0)],
            )
            for label in "ABCDEF"
        ]
        self.handler = SummarySegmentHandler()
        self.handler.set_source_videos(self.videos)
        self.criterion = ContentBasedRedundancy()
        self.segments = {video.name: video.segments[0] for video in self.videos}
        self.indices = {video.name: (index, 0) for index, video in enumerate(self.videos)}

    def edges(self, pairs):
        return [[self.indices[left], self.indices[right]] for left, right in pairs]

    def group(self, pairs, *, audit_report=None):
        return self.criterion.cluster_segments(
            self.handler, self.edges(pairs), self.videos, audit_report=audit_report,
        )

    def expected(self, labels):
        return {self.segments[label] for label in labels}

    def assert_disjoint(self, groups):
        self.assertTrue(all(isinstance(group, set) and group for group in groups))
        flattened = [segment for group in groups for segment in group]
        self.assertEqual(len(flattened), len(set(flattened)))

    def audit_report(self):
        ids = [self.handler.segment_id(video.segments[0]) for video in self.videos]
        return {
            "pair_decisions": [
                {
                    "pair_id": f"r{row}:c{column}",
                    "row_segment_id": left, "column_segment_id": right,
                    "cluster_ids": [], "exclusion_reasons": ["fixture_reason"],
                }
                for row, left in enumerate(ids)
                for column, right in enumerate(ids)
            ],
            "cluster_decisions": [],
        }

    def test_nacional_band_and_two_record_segments_form_one_indirect_group(self):
        videos = [
            Video("Nacional", "data/raw/nacional.mp4", segments=[VideoSegment("A", 0, 5, order=0)]),
            Video("Band", "data/raw/band.mp4", segments=[VideoSegment("B", 0, 5, order=0)]),
            Video("Record", "data/raw/record.mp4", segments=[
                VideoSegment("C", 10, 15, order=2), VideoSegment("D", 15, 20, order=3),
            ]),
        ]
        handler = SummarySegmentHandler()
        handler.set_source_videos(videos)
        before = [asdict(video) for video in videos]
        groups = self.criterion.cluster_segments(
            handler, [[(0, 0), (1, 0)], [(0, 0), (2, 0)], [(1, 0), (2, 1)]], videos,
        )
        self.assertEqual([{segment for video in videos for segment in video.segments}], groups)
        self.assert_disjoint(groups)
        self.assertEqual(before, [asdict(video) for video in videos])

    def test_disconnected_groups_follow_first_encounter_order(self):
        groups = self.group([("C", "D"), ("A", "B"), ("E", "F")])
        self.assertEqual([self.expected("CD"), self.expected("AB"), self.expected("EF")], groups)
        self.assert_disjoint(groups)

    def test_bridge_merges_existing_groups_and_compacts_group_order(self):
        groups = self.group([("A", "B"), ("C", "D"), ("E", "F"), ("B", "C")])
        self.assertEqual([self.expected("ABCD"), self.expected("EF")], groups)
        self.assert_disjoint(groups)

    def test_cycles_reversed_edges_and_duplicates_do_not_repeat_members(self):
        groups = self.group([("A", "B"), ("B", "A"), ("B", "C"), ("C", "A"), ("A", "B")])
        self.assertEqual([self.expected("ABC")], groups)
        self.assert_disjoint(groups)

    def test_edge_permutations_preserve_component_membership(self):
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("E", "F")]
        expected = {frozenset(self.expected("ABCD")), frozenset(self.expected("EF"))}
        for order in itertools.permutations(edges):
            with self.subTest(order=order):
                groups = self.group(order)
                self.assertEqual(expected, {frozenset(group) for group in groups})
                self.assert_disjoint(groups)

    def test_no_edges_returns_no_groups(self):
        self.assertEqual([], self.group([]))

    def test_long_reverse_chain_forms_one_group_without_recursive_traversal(self):
        segments = [VideoSegment(f"Chain {index}", index, index + 1, order=index) for index in range(2000)]
        videos = [
            Video("even", "data/raw/even.mp4", segments=segments[::2]),
            Video("odd", "data/raw/odd.mp4", segments=segments[1::2]),
        ]
        handler = SummarySegmentHandler(audit_enabled=False)
        handler.set_source_videos(videos)
        edges = [
            [((index + 1) % 2, (index + 1) // 2), (index % 2, index // 2)]
            for index in reversed(range(len(segments) - 1))
        ]
        groups = self.criterion.cluster_segments(handler, edges, videos)
        self.assertEqual([set(segments)], groups)
        self.assert_disjoint(groups)

    def test_discarded_and_output_segments_cannot_bridge_groups(self):
        for state in ("discard", "output"):
            for reverse in (False, True):
                with self.subTest(state=state, reverse=reverse):
                    handler = SummarySegmentHandler()
                    handler.set_source_videos(self.videos)
                    if state == "discard":
                        handler.discard_segment(self.segments["E"], "Introduction")
                    else:
                        handler.add_output_segment(self.segments["E"], "Introduction")
                    pairs = [("A", "B"), ("C", "D"), ("B", "E"), ("E", "C")]
                    if reverse:
                        pairs = [(right, left) for left, right in pairs]
                    groups = self.criterion.cluster_segments(handler, self.edges(pairs), self.videos)
                    self.assertEqual([self.expected("AB"), self.expected("CD")], groups)
                    self.assert_disjoint(groups)

    def test_included_segment_remains_eligible_to_bridge_groups(self):
        self.handler.include_segment(self.segments["E"], "fixture")
        groups = self.group([("A", "B"), ("C", "D"), ("B", "E"), ("E", "C")])
        self.assertEqual([self.expected("ABCDE")], groups)

    def test_all_skipped_edges_leave_no_empty_group(self):
        self.handler.discard_segment(self.segments["A"], "Introduction")
        self.handler.add_output_segment(self.segments["C"], "Introduction")
        self.assertEqual([], self.group([("A", "B"), ("B", "C"), ("C", "D")]))

    def test_audit_ids_and_memberships_refer_to_final_groups_after_bridge(self):
        report = self.audit_report()
        pairs = [("A", "B"), ("C", "D"), ("E", "F"), ("B", "C"), ("B", "A")]
        groups = self.group(pairs, audit_report=report)
        self.assertEqual([self.expected("ABCD"), self.expected("EF")], groups)
        self.assertEqual(
            ["create_cluster", "create_cluster", "create_cluster", "merge_clusters", "add_within_cluster"],
            [item["action"] for item in report["cluster_decisions"]],
        )
        final_ids = [f"{self.criterion.name}:cluster{index}" for index in range(2)]
        self.assertEqual(final_ids, [cluster["cluster_id"] for cluster in report["clusters"]])
        self.assertEqual(
            [final_ids[0], final_ids[0], final_ids[1], final_ids[0], final_ids[0]],
            [item["cluster_id"] for item in report["cluster_decisions"]],
        )
        memberships = report["segment_cluster_memberships"]
        for index, group in enumerate(groups):
            ids = {self.handler.segment_id(segment) for segment in group}
            self.assertEqual(ids, set(report["clusters"][index]["segment_ids"]))
            for segment_id in ids:
                self.assertEqual([final_ids[index]], memberships[segment_id])
        pair_lookup = {pair["pair_id"]: pair for pair in report["pair_decisions"]}
        for decision in report["cluster_decisions"]:
            pair = pair_lookup[decision["pair_id"]]
            self.assertEqual(
                [pair["row_segment_id"], pair["column_segment_id"]], decision["segment_ids"],
            )
            self.assertEqual([decision["cluster_id"]], pair["cluster_ids"])
        for pair in report["pair_decisions"]:
            shared = set(memberships[pair["row_segment_id"]]) & set(memberships[pair["column_segment_id"]])
            self.assertEqual(shared, set(pair["cluster_ids"]))
            self.assertEqual(["fixture_reason"], pair["exclusion_reasons"])

    def test_audit_skipped_edges_have_no_cluster_id_or_membership(self):
        self.handler.add_output_segment(self.segments["E"], "Introduction")
        report = self.audit_report()
        groups = self.group([("A", "B"), ("C", "D"), ("B", "E"), ("E", "C")], audit_report=report)
        self.assertEqual([self.expected("AB"), self.expected("CD")], groups)
        skipped = report["cluster_decisions"][2:]
        self.assertTrue(all(item["action"] == "skipped_ineligible" for item in skipped))
        self.assertTrue(all(item["cluster_id"] is None for item in skipped))
        self.assertNotIn(self.handler.segment_id(self.segments["E"]), report["segment_cluster_memberships"])
        for pair in report["pair_decisions"]:
            if self.handler.segment_id(self.segments["E"]) in (pair["row_segment_id"], pair["column_segment_id"]):
                self.assertEqual([], pair["cluster_ids"])
