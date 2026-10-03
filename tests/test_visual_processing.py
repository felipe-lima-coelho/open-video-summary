"""Exact visual equivalence and per-evaluation reuse, without model services."""

import json
import tempfile
import unittest
from math import log10
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from open_video_summary import __main__ as cli
from open_video_summary.core.selection_criteria.quality import QualityPick
from open_video_summary.entities.image import Keyframe
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.handlers.image import KeyframeHandler, _best_match_index
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.utils.config import PROJECT_DIR
from open_video_summary.utils.processing.image import BagOfVisualWords, ImageProcessor
from open_video_summary.utils.processing.metrics import (
    VisualProfile,
    collect_visual_profile,
    visual_count,
    visual_stage,
)
from open_video_summary.utils.processing.video import VideoProcessor


def legacy_num_matches(kf, other, threshold=0.95):
    """Reference algorithm before the equivalent comparison optimizations."""
    num_match = 0
    d1_t, d2_t = map(np.transpose, (kf.descriptor, other.descriptor))
    for i, desc in enumerate(kf.descriptor):
        sim = np.dot(desc, d2_t)
        self_match = np.argsort(-sim)[0]
        if sim[self_match] >= threshold:
            sim_check = np.dot(other.descriptor[self_match], d1_t)
            other_match = np.argsort(-sim_check)[0]
            num_match += (sim_check[other_match] >= threshold) and (other_match == i)
    return num_match


class DescriptorMatchingTests(unittest.TestCase):
    def test_best_match_preserves_sort_ties_nan_infinity_and_integer_overflow(self):
        cases = [
            np.array([0.0, 3.0, 1.0]),
            np.array([0.0] + [2.0] * 40 + [1.0]),
            np.zeros(40),
            np.array([-0.0, 0.0, -0.0, 0.0]),
            np.array([np.nan, 1.0, 4.0, np.nan]),
            np.full(40, np.nan),
            np.array([-np.inf, np.inf, 1.0]),
            np.array([np.inf, np.inf, 1.0]),
            np.array([0, 1, 255], dtype=np.uint8),
            np.array([-128, 127, 0], dtype=np.int8),
            np.array([1, 4, 2], dtype=object),
        ]
        for values in cases:
            with self.subTest(values=values):
                self.assertEqual(np.argsort(-values)[0], _best_match_index(values))
        with self.assertRaises(IndexError):
            _best_match_index(np.array([]))

    def test_matches_equal_reference_on_realistic_tied_and_nonfinite_descriptors(self):
        rng = np.random.default_rng(12)
        cases = [
            (
                rng.integers(0, 256, (23, 128)).astype(np.float32),
                rng.integers(0, 256, (19, 128)).astype(np.float32),
            ),
            (
                np.ones((30, 128), dtype=np.float32),
                np.ones((35, 128), dtype=np.float32),
            ),
            (
                np.zeros((20, 128), dtype=np.float32),
                np.zeros((25, 128), dtype=np.float32),
            ),
            (
                np.array([[np.nan, 1], [1, 2], [np.inf, 1]], dtype=np.float32),
                np.array([[1, 1], [np.nan, 1]], dtype=np.float32),
            ),
            (
                rng.normal(size=(18, 256)).astype(np.float32)[:, ::2],
                rng.normal(size=(24, 256)).astype(np.float32)[::2, ::2],
            ),
        ]
        for left, right in cases:
            for threshold in (-1.0, 0.0, 0.95, 100000000.0):
                with self.subTest(shape=(left.shape, right.shape), threshold=threshold):
                    keyframe, other = Keyframe(left), Keyframe(right)
                    self.assertEqual(
                        legacy_num_matches(keyframe, other, threshold),
                        KeyframeHandler.num_matches(keyframe, other, threshold),
                    )

    def test_repeated_reverse_comparison_is_reused_with_same_match_count(self):
        keyframe = Keyframe(np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
        other = Keyframe(np.array([[1.0, 0.0], [0.0, 0.0]]))
        expected = legacy_num_matches(keyframe, other)
        with patch("open_video_summary.handlers.image.dot", wraps=np.dot) as dot:
            self.assertEqual(expected, KeyframeHandler.num_matches(keyframe, other))
        self.assertEqual(4, dot.call_count)

    def test_empty_descriptors_keep_original_result_or_error(self):
        empty = Keyframe(np.empty((0, 128), dtype=np.float32))
        nonempty = Keyframe(np.ones((1, 128), dtype=np.float32))
        self.assertEqual(0, KeyframeHandler.num_matches(empty, nonempty))
        with self.assertRaises(IndexError):
            KeyframeHandler.num_matches(nonempty, empty)


class SiftAndVideoTests(unittest.TestCase):
    def test_detector_is_reused_and_only_original_inner_frames_are_processed(self):
        frames = [np.full((12, 12), value, dtype=np.uint8) for value in range(4)]
        descriptors = [np.full((2, 128), value, dtype=np.float32) for value in (1, 2)]
        detector = Mock()
        detector.detectAndCompute.side_effect = [(None, value) for value in descriptors]
        with (
            patch(
                "open_video_summary.utils.processing.image.SIFT_create",
                return_value=detector,
            ) as create,
            patch.object(KeyframeHandler, "is_keyframe", return_value=True),
        ):
            actual = ImageProcessor.ks_sift(frames)
        create.assert_called_once_with()
        self.assertEqual(2, detector.detectAndCompute.call_count)
        for call, frame in zip(detector.detectAndCompute.call_args_list, frames[1:-1]):
            self.assertIs(frame, call.args[0])
        np.testing.assert_array_equal(np.concatenate(descriptors), actual)

    def test_real_sift_reuse_preserves_descriptors_and_keyframes_exactly(self):
        rng = np.random.default_rng(4)
        frames = [rng.integers(0, 256, (96, 128), dtype=np.uint8) for _ in range(5)]
        selected = []
        for frame in frames[1:-1]:
            _, descriptor = cv2.SIFT_create().detectAndCompute(frame, None)
            if descriptor is not None:
                keyframe = Keyframe(descriptor)
                if not selected or all(
                    legacy_num_matches(keyframe, previous)
                    < 0.1 * previous.descriptor_size
                    for previous in selected
                ):
                    selected.append(keyframe)
        expected = np.concatenate([keyframe.descriptor for keyframe in selected])
        np.testing.assert_array_equal(expected, ImageProcessor.ks_sift(frames))

    def test_no_descriptors_preserves_concatenation_error(self):
        with self.assertRaises(ValueError):
            ImageProcessor.ks_sift([])
        frames = [np.zeros((24, 24), dtype=np.uint8)] * 3
        with self.assertRaises(ValueError):
            ImageProcessor.ks_sift(frames)

    def test_video_keeps_full_resolution_sampling_and_interval_semantics(self):
        for start, end, indices in ((0, None, [0, 25, 50]), (1.5, 2.0, [50])):
            images = [np.full((18, 32, 3), i, dtype=np.uint8) for i in range(54)]
            video = Mock()
            video.get.side_effect = [25.0, 54.0]
            video.read.side_effect = [(True, image) for image in images] + [
                (False, None)
            ]
            profile = VisualProfile()
            with (
                patch(
                    "open_video_summary.utils.processing.video.VideoCapture",
                    return_value=video,
                ),
                collect_visual_profile(profile),
            ):
                actual = VideoProcessor.retrieve_video_frames(
                    "source.mp4", grayscale=True, start_second=start, end_second=end
                )
            self.assertEqual(len(indices), len(actual))
            for index, frame in zip(indices, actual):
                np.testing.assert_array_equal(np.full((18, 32), index), frame)
            self.assertEqual(1, profile.counters["video_reads"])
            self.assertEqual(len(indices), profile.counters["sampled_frames"])
            video.release.assert_called_once_with()


class QualityCacheTests(unittest.TestCase):
    def setUp(self):
        output = PROJECT_DIR / "outputs"
        output.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=output)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.paths = [self.directory / f"source_{i}.mp4" for i in range(3)]
        for path in self.paths:
            path.write_bytes(b"fixture")
        self.descriptors = np.arange(256, dtype=np.float32).reshape(2, 128)

    def segment(self, source, index=0, relative=False):
        path = self.paths[source]
        if relative:
            path = path.relative_to(PROJECT_DIR)
        return VideoSegment(
            content=f"candidate {source}/{index}",
            start=float(index),
            end=index + 1.0,
            order=index,
            video_path=path.as_posix(),
        )

    def handler(self, groups):
        handler = SummarySegmentHandler()
        for group in groups:
            handler.add_segments_to_pick(set(group), "fixture")
        return handler

    def dataframe(self, items):
        return DataFrame({"word": [1.0] * len(items)}, index=list(items))

    def run_mocked(self, criterion, handler, dataframe=None):
        with (
            patch.object(
                VideoProcessor, "retrieve_video_frames", return_value=["frame"]
            ) as frames,
            patch.object(
                ImageProcessor, "ks_sift", return_value=self.descriptors
            ) as extract,
            patch.object(
                criterion, "get_bovw_dataframe", side_effect=dataframe or self.dataframe
            ) as bovw,
        ):
            criterion.features_extractor = extract
            criterion.evaluate(handler)
        return frames, extract, bovw

    def test_same_video_segments_and_path_aliases_reuse_only_descriptors(self):
        first = self.segment(0, 0)
        second = self.segment(0, 1, relative=True)
        third = self.segment(1, 2)
        handler = self.handler([[first, third], [second, third]])
        criterion = QualityPick(source_criteria="fixture")
        frames, extract, bovw = self.run_mocked(criterion, handler)
        self.assertEqual(2, frames.call_count)
        self.assertEqual(2, extract.call_count)
        self.assertEqual(2, bovw.call_count)
        self.assertEqual(list(handler.pick[0]), list(bovw.call_args_list[0].args[0]))
        self.assertEqual(list(handler.pick[1]), list(bovw.call_args_list[1].args[0]))
        self.assertEqual(
            {first, second, third},
            set().union(*[set(call.args[0]) for call in bovw.call_args_list]),
        )
        self.assertEqual(2, criterion.last_profile["counters"]["cache_hits"])
        self.assertEqual(2, criterion.last_profile["counters"]["cache_misses"])
        for call in frames.call_args_list:
            self.assertEqual({"grayscale": True}, call.kwargs)

    def test_cache_is_fresh_on_each_evaluation_and_absent_on_direct_extraction(self):
        criterion = QualityPick(source_criteria="fixture")
        handler = self.handler([[self.segment(0)]])
        for _ in range(2):
            _, extract, _ = self.run_mocked(criterion, handler)
            self.assertEqual(1, extract.call_count)
            self.assertEqual(1, criterion.last_profile["counters"]["cache_misses"])
        with (
            patch.object(VideoProcessor, "retrieve_video_frames", return_value=[]),
            patch.object(
                ImageProcessor, "ks_sift", return_value=self.descriptors
            ) as extract,
        ):
            criterion.features_extractor = extract
            for _ in range(2):
                criterion.extract_segments_visual_features({self.segment(0)})
        self.assertEqual(2, extract.call_count)

    def test_file_change_invalidates_cached_descriptors_within_run(self):
        criterion = QualityPick(source_criteria="fixture")
        handler = self.handler([[self.segment(0, 0)], [self.segment(0, 1)]])

        def dataframe(items):
            self.paths[0].write_bytes(self.paths[0].read_bytes() + b"changed")
            return self.dataframe(items)

        _, extract, _ = self.run_mocked(criterion, handler, dataframe)
        self.assertEqual(2, extract.call_count)
        self.assertEqual(2, criterion.last_profile["counters"]["cache_misses"])

    def test_memory_limit_evicts_or_bypasses_without_changing_features(self):
        groups = [[self.segment(0, 0)], [self.segment(1, 0)], [self.segment(0, 1)]]
        for limit in (0, self.descriptors.nbytes - 1, self.descriptors.nbytes):
            criterion = QualityPick("fixture", max_descriptor_cache_bytes=limit)
            _, extract, bovw = self.run_mocked(criterion, self.handler(groups))
            self.assertEqual(3, extract.call_count)
            for call in bovw.call_args_list:
                for descriptors in call.args[0].values():
                    np.testing.assert_array_equal(self.descriptors, descriptors)
            self.assertLessEqual(
                criterion.last_profile["counters"].get("cache_peak_bytes", 0), limit
            )
            if limit == self.descriptors.nbytes:
                self.assertEqual(
                    2, criterion.last_profile["counters"]["cache_evictions"]
                )

    def test_custom_extractor_keeps_per_segment_call_behavior(self):
        extract = Mock(side_effect=[self.descriptors, self.descriptors + 1])
        criterion = QualityPick("fixture", features_extractor=extract)
        with (
            patch.object(VideoProcessor, "retrieve_video_frames", return_value=[]),
            patch.object(criterion, "get_bovw_dataframe", side_effect=self.dataframe),
        ):
            criterion.evaluate(
                self.handler([[self.segment(0, 0)], [self.segment(0, 1)]])
            )
        self.assertEqual(2, extract.call_count)
        self.assertEqual(2, criterion.last_profile["counters"]["cache_bypasses"])

    def test_failed_run_does_not_leave_cache_or_active_profile(self):
        criterion = QualityPick("fixture")
        handler = self.handler([[self.segment(0)]])
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            self.run_mocked(
                criterion, handler, Mock(side_effect=RuntimeError("fixture failure"))
            )
        self.assertEqual("failed", criterion.last_profile["status"])
        failed_counters = dict(criterion.last_profile["counters"])
        visual_count("outside_run")
        self.assertEqual(failed_counters, criterion.last_profile["counters"])
        _, extract, _ = self.run_mocked(criterion, handler)
        self.assertEqual(1, extract.call_count)
        self.assertEqual("completed", criterion.last_profile["status"])


class VisualProfileAndBovwTests(unittest.TestCase):
    def test_nested_substages_do_not_double_count(self):
        profile = VisualProfile()
        with (
            patch(
                "open_video_summary.utils.processing.metrics.perf_counter",
                side_effect=[0.0, 1.0, 2.0, 5.0, 8.0, 10.0],
            ),
            collect_visual_profile(profile),
        ):
            with visual_stage("parent"):
                with visual_stage("child"):
                    pass
        self.assertEqual({"parent": 4.0, "child": 3.0}, dict(profile.stage_seconds))
        self.assertEqual(10.0, profile.total_seconds)

    def test_profile_is_separate_from_exclusive_stages_and_restored_when_nested(self):
        outer, inner = VisualProfile(), VisualProfile()
        with collect_visual_profile(outer):
            with visual_stage("first"):
                visual_count("outer")
            with collect_visual_profile(inner):
                visual_count("inner")
            visual_count("outer")
        self.assertEqual({"outer": 2}, dict(outer.counters))
        self.assertEqual({"inner": 1}, dict(inner.counters))
        self.assertLessEqual(sum(outer.stage_seconds.values()), outer.total_seconds)
        self.assertEqual("completed", inner.status)

    def test_bovw_keeps_descriptor_multiplicity_order_default_kmeans_and_weights(self):
        first = VideoSegment("first", 0, 1)
        second = VideoSegment("second", 1, 2)
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        items = {second: features, first: features}
        kmeans = Mock()
        kmeans.predict.side_effect = [np.array([2, 2, 1]), np.array([1, 0, 1])]
        profile = VisualProfile()
        with (
            patch(
                "open_video_summary.utils.processing.image.KMeans", return_value=kmeans
            ) as create,
            collect_visual_profile(profile),
        ):
            bovw = BagOfVisualWords(items, dict_size=300)
            bovw.fit_kmeans()
            actual = bovw.generate_bovw_dataframe()
        create.assert_called_once_with(n_clusters=300)
        np.testing.assert_array_equal(
            np.concatenate([features, features]), kmeans.fit.call_args.args[0]
        )
        expected = DataFrame(
            {
                2: [2 * log10(300), np.nan],
                1: [log10(150), 2 * log10(150)],
                0: [np.nan, log10(300)],
            },
            index=[second, first],
        )
        expected.index.name = "segment"
        expected.columns = expected.columns.astype(object)
        assert_frame_equal(expected, actual, check_exact=True)
        self.assertEqual(1, profile.counters["kmeans_fits"])
        self.assertEqual(2, profile.counters["kmeans_predictions"])
        self.assertLessEqual(sum(profile.stage_seconds.values()), profile.total_seconds)

    def test_empty_bovw_input_preserves_concatenation_error(self):
        with self.assertRaises(ValueError):
            BagOfVisualWords({}, dict_size=300).fit_kmeans()

    def test_cli_writes_visual_profile_without_changing_summary_metadata(self):
        output_root = PROJECT_DIR / "outputs"
        output_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_root) as directory:
            directory = Path(directory)
            source = directory / "source.mp4"
            source.write_bytes(b"fixture")
            video = Video("source", str(source), segments=[VideoSegment("text", 0, 1)])
            profile = {
                "status": "completed",
                "total_seconds": 0.5,
                "counters": {"cache_hits": 2},
            }
            summary = Video(
                "summary", str(directory / "summary.mp4"), segments=video.segments
            )
            summarizer = SimpleNamespace(
                summarize=Mock(return_value=summary),
                selection_criteria=[
                    SimpleNamespace(name="QualityPick", last_profile=profile)
                ],
            )
            args = SimpleNamespace(
                dataset="fixture.json",
                output=summary.path,
                title=summary.name,
                no_render=True,
                threads=2,
            )
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "open_video_summary.core.summarizers": SimpleNamespace(
                            HSMVideoSumm=summarizer
                        ),
                        "torch": SimpleNamespace(set_num_threads=Mock()),
                    },
                ),
                patch(
                    "open_video_summary.parsers.video.VideoLoader.load_videos_from_json",
                    return_value=[video],
                ),
                patch(
                    "open_video_summary.parsers.video.VideoDumper.dump_videos_to_json"
                ) as dump,
                patch(
                    "open_video_summary.__main__.ModelPaths.SUBJECTIVITY_CLASSIFIER",
                    str(directory),
                ),
                patch("cv2.setNumThreads"),
            ):
                cli._summarize(args)
            saved = json.loads(
                (directory / "summary_visual_profile.json").read_text(encoding="utf-8")
            )
            self.assertEqual(profile, saved)
            self.assertEqual([summary], dump.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
