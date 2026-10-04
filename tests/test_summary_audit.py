"""Complete numerical coverage, observational selection logging, and persistence."""

import contextlib
import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from pandas import DataFrame, MultiIndex
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from open_video_summary import __main__ as cli
from open_video_summary.core.selection_criteria.quality import QualityPick
from open_video_summary.core.selection_criteria.redundancy import ContentBasedRedundancy
from open_video_summary.core.selection_criteria.chronology import ClusterBasedChronology
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.handlers.summary import SummarySegmentHandler, SummarySegmentHandlerIO
from open_video_summary.utils.audit import finalize_summary_audit, save_summary_audit
from open_video_summary.utils.config import PROJECT_DIR
from open_video_summary.utils.helpers import custom_cosine
from open_video_summary.utils.processing.image import ImageProcessor
from open_video_summary.utils.processing.video import VideoProcessor


# Load the library coordinator without importing the package's eager pretrained
# HSMVideoSumm singleton. These tests exercise its real file with no model load.
_spec = importlib.util.spec_from_file_location(
    "audit_test_summarizer", PROJECT_DIR / "open_video_summary/core/summarizers/base.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
Summarizer = _module.Summarizer


def fixture_videos():
    return [
        Video(
            f"source_{video}", f"data/raw/fixture_{video}.mp4",
            segments=[
                VideoSegment(
                    f"Private transcript {video}/{segment} shared words", segment * 5, (segment + 1) * 5,
                    order=segment, video_topic="topic", global_topic="global",
                )
                for segment in range(2)
            ],
        )
        for video in range(3)
    ]


def fixture_bow():
    vectors = np.array([
        [1.0, 0.0, 0.0], [0.0, 0.1, 0.1],
        [0.9, 0.4, 0.0], [0.0, 0.0, 0.2],
        [0.8, 0.0, 0.0], [0.7, 0.8, 0.0],
    ])
    index = MultiIndex.from_tuples(
        [(video, segment) for video in range(3) for segment in range(2)],
        names=["video_index", "segment_index"],
    )
    return DataFrame(vectors, index=index)


class CorrelationAuditTests(unittest.TestCase):
    def make_handler(self, enabled=True):
        handler = SummarySegmentHandler(audit_enabled=enabled)
        handler.set_source_videos(fixture_videos())
        return handler

    def run_criterion(self, handler, threshold=0.1):
        criterion = ContentBasedRedundancy(reference_time_sec=30, base_threshold=threshold)
        with patch.object(criterion, "get_bow_df", return_value=fixture_bow()):
            criterion.evaluate(handler)
        return criterion

    def test_all_computed_pairs_retained_with_structural_diagonal_label(self):
        handler = self.make_handler()
        with patch(
            "open_video_summary.core.selection_criteria.redundancy.custom_cosine", wraps=custom_cosine
        ) as similarity:
            criterion = self.run_criterion(handler)
        report = criterion.last_audit
        self.assertEqual(15, similarity.call_count)
        self.assertEqual(36, len(report["pair_decisions"]))
        self.assertEqual(15, sum(pair["origin"] == "computed" for pair in report["pair_decisions"]))
        expected = fixture_bow().to_numpy() @ fixture_bow().to_numpy().T
        np.fill_diagonal(expected, 1)
        np.testing.assert_array_equal(expected, np.array(report["raw_matrix"]))
        diagonal = report["pair_decisions"][3 * 6 + 3]
        self.assertEqual("pandas_diagonal", diagonal["origin"])
        self.assertEqual(1, diagonal["value"])
        self.assertNotEqual(np.dot(fixture_bow().iloc[3], fixture_bow().iloc[3]), diagonal["value"])
        same_video = report["pair_decisions"][1]
        self.assertEqual("computed", same_video["origin"])
        self.assertEqual(0, same_video["value"])
        self.assertIn("same_video", same_video["exclusion_reasons"])
        below = report["pair_decisions"][1 * 6 + 2]
        self.assertGreater(below["value"], 0)
        self.assertIn("not_strictly_above_threshold", below["exclusion_reasons"])
        maximum = next(pair for pair in report["pair_decisions"] if pair["row"] == 2 and pair["column"] == 5)
        self.assertEqual(0.1, report["threshold"])
        self.assertEqual(">", report["threshold_operator"])
        self.assertTrue(maximum["is_video_pair_maximum"])
        self.assertEqual(maximum["value"], maximum["video_pair_maximum"])
        lower = report["pair_decisions"][2 * 6]
        self.assertEqual("mirrored", lower["origin"])
        self.assertIn("lower_triangle_positive_mask", lower["exclusion_reasons"])
        self.assertNotIn("Private transcript", json.dumps(handler.audit))

    def test_maximum_then_eligibility_preserves_exclusions_and_overlap_behavior(self):
        handler = self.make_handler()
        criterion = self.run_criterion(handler)
        self.assertEqual(2, len(handler.pick))
        repeated = handler.source[1].segments[0]
        self.assertTrue(all(repeated in cluster for cluster in handler.pick))
        self.assertEqual(2, len(criterion.last_audit["segment_cluster_memberships"]["v1:s0"]))
        self.assertEqual(["create_cluster", "add_to_row_segment_cluster", "create_cluster"], [
            item["action"] for item in criterion.last_audit["cluster_decisions"]
        ])

        excluded = self.make_handler()
        excluded.discard_segment(excluded.source[1].segments[0], "Introduction")
        excluded.add_output_segment(excluded.source[0].segments[1], "Introduction")
        report = self.run_criterion(excluded).last_audit
        self.assertEqual(1, len(excluded.pick))
        skipped = [item for item in report["cluster_decisions"] if item["action"] == "skipped_ineligible"]
        self.assertEqual(2, len(skipped))
        pair = next(item for item in report["pair_decisions"] if item["row"] == 0 and item["column"] == 2)
        self.assertTrue(pair["retained_after_filters"])
        self.assertTrue(pair["is_video_pair_maximum"])
        self.assertFalse(pair["eligible_for_clustering"])
        self.assertIn("column_already_discarded", pair["exclusion_reasons"])
        self.assertEqual([], pair["cluster_ids"])
        self.assertEqual(["Introduction"], report["segments"][2]["discarded_by"])

    def test_threshold_is_strict_and_maximum_is_after_filters(self):
        report = self.run_criterion(self.make_handler(), threshold=0.9).last_audit
        equal = next(item for item in report["pair_decisions"] if item["row"] == 0 and item["column"] == 2)
        self.assertEqual(0.9, equal["value"])
        self.assertFalse(equal["retained_after_filters"])
        self.assertIn("not_strictly_above_threshold", equal["exclusion_reasons"])
        self.assertIsNone(equal["video_pair_maximum"])

    def test_all_exact_video_pair_maximum_ties_are_recorded(self):
        handler = self.make_handler()
        bow = fixture_bow()
        bow.iloc[5] = bow.iloc[4]
        criterion = ContentBasedRedundancy(reference_time_sec=30, base_threshold=0.1)
        with patch.object(criterion, "get_bow_df", return_value=bow):
            criterion.evaluate(handler)
        maxima = [
            pair for pair in criterion.last_audit["pair_decisions"]
            if pair["video_pair"] == [0, 2] and pair["is_video_pair_maximum"]
        ]
        self.assertEqual(2, len(maxima))
        self.assertEqual([0.8, 0.8], [pair["value"] for pair in maxima])

    def test_unavailable_matrix_values_are_explicit_and_finite_json(self):
        handler = self.make_handler()
        bow = fixture_bow()
        bow.iloc[5] = np.nan
        criterion = ContentBasedRedundancy(reference_time_sec=30, base_threshold=0.1)
        with patch.object(criterion, "get_bow_df", return_value=bow):
            criterion.evaluate(handler)
        report = criterion.last_audit
        self.assertIsNone(report["raw_matrix"][0][5])
        pair = report["pair_decisions"][5]
        self.assertEqual("nan", pair["value_status"])
        self.assertEqual("unavailable", pair["origin"])
        self.assertFalse(pair["retained_after_filters"])
        self.assertIn("nonfinite_similarity", pair["exclusion_reasons"])
        json.dumps(report, allow_nan=False)

    def test_audit_does_not_change_legacy_filters_grouping_or_rng(self):
        outcomes = []
        for enabled in (False, True):
            handler = self.make_handler(enabled)
            random.seed(911)
            np.random.seed(911)
            self.run_criterion(handler)
            outcomes.append((handler.pick, random.getstate(), np.random.get_state()))
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        self.assertEqual(outcomes[0][1], outcomes[1][1])
        np.testing.assert_array_equal(outcomes[0][2][1], outcomes[1][2][1])
        self.assertEqual(outcomes[0][2][2:], outcomes[1][2][2:])

    def test_tfidf_includes_all_segments_and_runs_once(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        handler = self.make_handler()
        handler.discard_segment(handler.source[0].segments[0], "Introduction")
        vectorizer = Mock(wraps=TfidfVectorizer(use_idf=True, smooth_idf=False))
        with patch(
            "open_video_summary.core.selection_criteria.redundancy.TfidfVectorizer", return_value=vectorizer
        ) as create:
            criterion = ContentBasedRedundancy()
            criterion.evaluate(handler)
        create.assert_called_once_with(use_idf=True, smooth_idf=False)
        vectorizer.fit_transform.assert_called_once()
        self.assertEqual(6, len(vectorizer.fit_transform.call_args.args[0]))
        self.assertEqual(6, len(criterion.last_audit["raw_matrix"]))


class QualityAuditTests(unittest.TestCase):
    def make_handler(self, enabled=True):
        handler = SummarySegmentHandler(audit_enabled=enabled)
        videos = fixture_videos()
        handler.set_source_videos(videos)
        group = {videos[0].segments[0], videos[1].segments[0], videos[2].segments[0]}
        handler.add_segments_to_pick(group, "fixture")
        return handler

    def test_individual_weights_scores_order_rank_ties_and_losers_are_observational(self):
        handler = self.make_handler()
        group = handler.pick[0]
        features = {segment: np.ones((3, 128), dtype=np.float32) for segment in group}
        ordered = list(group)
        frame = DataFrame({0: [2.0, 1.0, np.nan], 1: [1.0, 2.0, 1.0]}, index=ordered)
        criterion = QualityPick("fixture", bovw_dict_size=3)
        with (
            patch.object(criterion, "extract_segments_visual_features", return_value=features) as extract,
            patch.object(criterion, "get_bovw_dataframe", return_value=frame),
        ):
            criterion.evaluate(handler)
        extract.assert_called_once()
        report = criterion.last_audit["clusters"][0]
        ids = [handler.segment_id(segment) for segment in ordered]
        self.assertEqual(ids, [item["segment_id"] for item in report["candidates"]])
        self.assertEqual(ids, report["ranking_segment_ids"])
        self.assertEqual([ids[0]], report["chosen_segment_ids"])
        self.assertEqual([3.0, 3.0, 1.0], [item["score"] for item in report["candidates"]])
        self.assertEqual([1, 2, 3], [item["rank"] for item in report["candidates"]])
        self.assertEqual([True, False, False], [item["chosen"] for item in report["candidates"]])
        self.assertEqual([{"score": 3.0, "segment_ids": ids[:2]}], report["ties"])
        self.assertIsNone(report["candidates"][2]["visual_word_weights"][0]["weight"])
        self.assertEqual({ordered[0]}, handler.include)
        self.assertEqual(set(), handler.discard)
        self.assertEqual([], handler.agent_logs["QualityPick"].discard)

    def test_logging_does_not_add_kmeans_or_feature_calls_or_consume_random_state(self):
        rng = np.random.default_rng(991)
        arrays = [rng.normal(size=(20, 128)).astype(np.float32) for _ in range(3)]
        states, handlers = [], []
        with threadpool_limits(limits=1), patch(
            "open_video_summary.utils.processing.image.KMeans", wraps=KMeans
        ) as create:
            for enabled in (False, True):
                handler = self.make_handler(enabled)
                ordered = list(handler.pick[0])
                extract = Mock(side_effect=arrays)
                criterion = QualityPick("fixture", bovw_dict_size=3, features_extractor=extract)
                random.seed(942)
                np.random.seed(942)
                with patch.object(VideoProcessor, "retrieve_segment_frames", return_value=["frame"]):
                    criterion.evaluate(handler)
                self.assertEqual(3, extract.call_count)
                self.assertEqual(ordered, list(handler.pick[0]))
                handlers.append(handler)
                states.append((random.getstate(), np.random.get_state()))
        self.assertEqual(2, create.call_count)
        self.assertEqual(handlers[0].include, handlers[1].include)
        self.assertEqual(handlers[0].agent_logs, handlers[1].agent_logs)
        self.assertEqual(states[0][0], states[1][0])
        np.testing.assert_array_equal(states[0][1][1], states[1][1][1])
        self.assertEqual(states[0][1][2:], states[1][1][2:])
        self.assertEqual({}, handlers[0].audit["criteria"])

    def test_zero_descriptor_candidate_scores_zero_and_vocabulary_is_preserved(self):
        handler = self.make_handler()
        ordered = list(handler.pick[0])
        rng = np.random.default_rng(33)
        features = {
            ordered[0]: np.empty((0, 128), dtype=np.float32),
            ordered[1]: rng.normal(size=(5, 128)).astype(np.float32),
            ordered[2]: rng.normal(size=(5, 128)).astype(np.float32),
        }
        criterion = QualityPick("fixture", bovw_dict_size=3)
        with threadpool_limits(limits=1), patch.object(
            criterion, "extract_segments_visual_features", return_value=features
        ):
            criterion.evaluate(handler)
        candidate = criterion.last_audit["clusters"][0]["candidates"][0]
        self.assertEqual("no_descriptors", candidate["descriptor_status"])
        self.assertEqual(0, candidate["score"])
        self.assertEqual(3, criterion.last_audit["metric"]["dictionary_size"])


class AuditPersistenceTests(unittest.TestCase):
    def setUp(self):
        output = PROJECT_DIR / "outputs"
        output.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=output)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def test_library_default_sidecar_and_handler_roundtrip_are_portable_and_old_input_loads(self):
        videos = fixture_videos()
        summarizer = Summarizer([])
        output = self.directory / "summary.mp4"
        handler_path = self.directory / "handler.json"
        summarizer.summarize(videos, video_output_path=str(output), handler_output_path=str(handler_path))
        audit_path = self.directory / "summary_audit.json"
        report = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(1, report["schema_version"])
        self.assertEqual("completed", report["status"])
        self.assertEqual("data/raw/fixture_0.mp4", report["sources"][0]["path"])
        self.assertEqual("data/raw/fixture_0.mp4", report["segments"][0]["video_path"])
        self.assertEqual("v0:s0", report["segments"][0]["segment_id"])
        self.assertEqual(0, report["segments"][0]["order"])
        self.assertEqual(0, report["segments"][0]["start"])
        self.assertEqual(5, report["segments"][0]["end"])
        self.assertNotIn("Private transcript", audit_path.read_text(encoding="utf-8"))
        restored = SummarySegmentHandlerIO.load(str(handler_path))
        resaved = self.directory / "resaved.json"
        SummarySegmentHandlerIO.save(restored, str(resaved))
        self.assertEqual(json.loads(handler_path.read_text()), json.loads(resaved.read_text()))

        old_data = json.loads(handler_path.read_text(encoding="utf-8"))
        old_data.pop("_SummarySegmentHandler__audit")
        old_data.pop("audit_enabled")
        old_path = self.directory / "old.json"
        old_path.write_text(json.dumps(old_data), encoding="utf-8")
        old = SummarySegmentHandlerIO.load(str(old_path))
        self.assertEqual(restored.source, old.source)
        self.assertTrue(old.audit_enabled)
        self.assertEqual(6, len(old.audit["segments"]))

    def test_save_output_false_keeps_audit_in_memory_and_disabled_capture_is_empty(self):
        summarizer = Summarizer([])
        output = self.directory / "memory.mp4"
        summarizer.summarize(fixture_videos(), video_output_path=str(output), save_output=False)
        self.assertEqual("completed", summarizer.last_audit["status"])
        self.assertFalse((self.directory / "memory_audit.json").exists())
        summarizer.summarize(fixture_videos(), video_output_path=str(output), save_output=False, collect_audit=False)
        self.assertIsNone(summarizer.last_audit)
        self.assertEqual([], summarizer.last_handler.audit["segments"])
        summarizer.last_handler.segment_id(VideoSegment("external", 0, 1))
        self.assertEqual([], summarizer.last_handler.audit["segments"])

    def test_serialization_is_strict_finite_and_preserves_outside_paths(self):
        output = self.directory / "finite.json"
        with tempfile.TemporaryDirectory() as elsewhere:
            outside = Path(elsewhere) / "outside.mp4"
            save_summary_audit({
                "score": np.float64(float("nan")), "weight": float("inf"),
                "count": np.int64(3), "video_path": str(outside),
            }, str(output))
            data = json.loads(output.read_text(), parse_constant=lambda value: self.fail(value))
            self.assertEqual({"score": None, "weight": None, "count": 3, "video_path": outside.as_posix()}, data)

    def test_insufficient_interval_descriptors_fail_with_persisted_partial_audit(self):
        class FixedGroups:
            name = "fixture"
            def evaluate(self, handler):
                handler.add_segments_to_pick({handler.source[0].segments[0]}, "fixture")
                return handler
        quality = QualityPick("fixture")
        summarizer = Summarizer([FixedGroups(), quality])
        output = self.directory / "failed.mp4"
        with patch.object(VideoProcessor, "retrieve_segment_frames", return_value=[]), self.assertRaisesRegex(ValueError, "needs 300 descriptors"):
            summarizer.summarize(fixture_videos(), video_output_path=str(output), handler_output_path=str(self.directory / "failed_handler.json"))
        report = json.loads((self.directory / "failed_audit.json").read_text())
        self.assertEqual("failed", report["status"])
        cluster = report["criteria"]["QualityPick"]["clusters"][0]
        self.assertEqual("insufficient_descriptors_for_dictionary", cluster["failure_reason"])
        self.assertEqual(0, cluster["candidates"][0]["descriptor_count"])
        self.assertEqual("no_sampled_frames", cluster["candidates"][0]["extraction_status"])
        self.assertIsNone(cluster["candidates"][0]["score"])

    def test_invalid_interval_failure_is_explicit_in_saved_audit(self):
        video = Video("invalid", "data/raw/invalid.mp4", segments=[VideoSegment("private", 0, 0)])
        class FixedGroups:
            def evaluate(self, handler):
                handler.add_segments_to_pick(set(handler.source[0].segments), "fixture")
                return handler
        summarizer = Summarizer([FixedGroups(), QualityPick("fixture")])
        output = self.directory / "invalid.mp4"
        with self.assertRaisesRegex(ValueError, "0 <= start < end"):
            summarizer.summarize([video], video_output_path=str(output))
        report = json.loads((self.directory / "invalid_audit.json").read_text())
        cluster = report["criteria"]["QualityPick"]["clusters"][0]
        self.assertEqual("invalid_interval", cluster["failure_reason"])
        self.assertEqual("invalid", cluster["candidates"][0]["interval_status"])
        self.assertIsNone(cluster["candidates"][0]["score"])

    def test_all_final_exclusions_are_traceable_to_existing_stage_or_visual_decisions(self):
        videos = fixture_videos()
        redundancy = ContentBasedRedundancy(reference_time_sec=30, base_threshold=0.1)
        quality = QualityPick("ContentBasedRedundancy", bovw_dict_size=3)
        summarizer = Summarizer([redundancy, quality, ClusterBasedChronology("ContentBasedRedundancy")])
        with (
            patch.object(redundancy, "get_bow_df", return_value=fixture_bow()),
            patch.object(quality, "extract_segments_visual_features", side_effect=lambda segments: {segment: np.ones((3, 128)) for segment in segments}),
            patch.object(quality, "get_bovw_dataframe", side_effect=lambda items: DataFrame({0: list(range(1, len(items) + 1))}, index=list(items))),
        ):
            summary = summarizer.summarize(videos, save_output=False)
        self.assertGreater(len(summary.segments), 0)
        for outcome in summarizer.last_audit["outcome"]["segments"]:
            if not outcome["output_positions"]:
                self.assertTrue(outcome["exclusion_reasons"], outcome)

    def test_cli_no_render_persists_audit_profile_metadata_and_scope(self):
        source_path = self.directory / "source.mp4"
        source_path.write_bytes(b"fixture")
        video = Video("source", str(source_path), segments=[VideoSegment("private text", 0, 5)])
        class FixedGroups:
            name = "fixture"
            def evaluate(self, handler):
                handler.add_segments_to_pick(set(handler.source[0].segments), "fixture")
                return handler
        quality = QualityPick("fixture", bovw_dict_size=3)
        summarizer = Summarizer([FixedGroups(), quality])
        output = self.directory / "cli.mp4"
        args = SimpleNamespace(dataset="fixture.json", output=str(output), title="fixture", no_render=True, threads=1, visual_scope="video")
        with (
            patch.dict("sys.modules", {
                "open_video_summary.core.summarizers": SimpleNamespace(HSMVideoSumm=summarizer),
                "torch": SimpleNamespace(set_num_threads=Mock()),
            }),
            patch("open_video_summary.parsers.video.VideoLoader.load_videos_from_json", return_value=[video]),
            patch("open_video_summary.__main__.ModelPaths.SUBJECTIVITY_CLASSIFIER", str(self.directory)),
            patch("cv2.setNumThreads"),
            patch.object(VideoProcessor, "retrieve_video_frames", return_value=["frame"]),
            patch.object(ImageProcessor, "ks_sift", return_value=np.ones((3, 128))) as extract,
            patch.object(quality, "get_bovw_dataframe", side_effect=lambda items: DataFrame({0: [1.0]}, index=list(items))),
            patch("open_video_summary.parsers.video.SummaryWriter.write_video_summary") as render,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            quality.features_extractor = extract
            cli._summarize(args)
        render.assert_not_called()
        self.assertFalse(output.exists())
        for suffix in (".json", "_handler.json", "_audit.json", "_visual_profile.json"):
            path = output.with_suffix(".json") if suffix == ".json" else output.with_name(f"{output.stem}{suffix}")
            self.assertTrue(path.is_file(), suffix)
        report = json.loads((self.directory / "cli_audit.json").read_text())
        self.assertEqual("video", report["criteria"]["QualityPick"]["scope"])
        candidate = report["criteria"]["QualityPick"]["clusters"][0]["candidates"][0]
        self.assertEqual("video", candidate["scope"])
        self.assertIsNone(candidate["start"])
        self.assertIsNone(candidate["end"])
        self.assertEqual(0, report["segments"][0]["start"])
        self.assertEqual(5, report["segments"][0]["end"])
        self.assertEqual("video", json.loads((self.directory / "cli_visual_profile.json").read_text())["settings"]["scope"])


if __name__ == "__main__":
    unittest.main()
