"""The same source interval is reusable; distinct intervals stay independent."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from pandas import DataFrame

from open_video_summary.core.selection_criteria.quality import QualityPick
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.utils.config import PROJECT_DIR
from open_video_summary.utils.processing.image import BagOfVisualWords, ImageProcessor
from open_video_summary.utils.processing.video import VideoProcessor
from threadpoolctl import threadpool_limits


class QualityScopeTests(unittest.TestCase):
    def setUp(self):
        root = PROJECT_DIR / "outputs"
        root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.paths = [self.directory / f"source_{index}.mp4" for index in range(2)]
        for path in self.paths:
            path.write_bytes(b"fixture")
        self.first = VideoSegment("first", 0, 5, order=0, video_path=str(self.paths[0]))
        self.alias = VideoSegment("alias", 0, 5, order=1, video_path=self.paths[0].relative_to(PROJECT_DIR).as_posix())
        self.later = VideoSegment("later", 5, 10, order=2, video_path=str(self.paths[0]))
        self.other = VideoSegment("other", 0, 5, order=0, video_path=str(self.paths[1]))

    def handler(self):
        handler = SummarySegmentHandler()
        handler.set_source_videos([
            Video("first source", str(self.paths[0]), segments=[self.first, self.alias, self.later]),
            Video("other source", str(self.paths[1]), segments=[self.other]),
        ])
        for group in ({self.first, self.other}, {self.alias, self.other}, {self.later, self.other}):
            handler.add_segments_to_pick(group, "fixture")
        return handler

    def frame(self, items):
        return DataFrame({0: [1.0] * len(items)}, index=list(items))

    def test_interval_cache_and_full_video_cache_have_correct_scope_and_provenance(self):
        for scope, misses, hits in (("segment", 3, 3), ("video", 2, 4)):
            with self.subTest(scope=scope):
                criterion = QualityPick("fixture", visual_scope=scope)
                with (
                    patch.object(VideoProcessor, "retrieve_segment_frames", side_effect=lambda path, start, end, **kwargs: [start]) as intervals,
                    patch.object(VideoProcessor, "retrieve_video_frames", return_value=[99]) as videos,
                    patch.object(ImageProcessor, "ks_sift", side_effect=lambda frames, **kwargs: np.full((3, 128), frames[0], dtype=np.float32)) as extract,
                    patch.object(criterion, "get_bovw_dataframe", side_effect=self.frame) as bovw,
                ):
                    criterion.features_extractor = extract
                    handler = self.handler()
                    criterion.evaluate(handler)
                self.assertEqual(misses, extract.call_count)
                self.assertEqual(misses, criterion.last_profile["counters"]["cache_misses"])
                self.assertEqual(hits, criterion.last_profile["counters"]["cache_hits"])
                self.assertEqual(misses if scope == "segment" else 0, intervals.call_count)
                self.assertEqual(misses if scope == "video" else 0, videos.call_count)
                features = {}
                for call in bovw.call_args_list:
                    features.update(call.args[0])
                np.testing.assert_array_equal(features[self.first], features[self.alias])
                if scope == "segment":
                    self.assertFalse(np.array_equal(features[self.first], features[self.later]))
                else:
                    np.testing.assert_array_equal(features[self.first], features[self.later])
                for cluster in criterion.last_audit["clusters"]:
                    for candidate in cluster["candidates"]:
                        self.assertEqual(scope, candidate["scope"])
                        self.assertEqual("not_used" if scope == "video" else "valid", candidate["interval_status"])
                        if scope == "video":
                            self.assertIsNone(candidate["start"])
                            self.assertIsNone(candidate["end"])

    def test_parallel_request_identity_includes_interval_and_deduplicates_alias(self):
        criterion = QualityPick("fixture", visual_threads=2)
        workers = MagicMock()
        workers.get.return_value = np.ones((3, 128), dtype=np.float32)
        with (
            patch("open_video_summary.core.selection_criteria.quality.DescriptorWorkers", return_value=workers) as create,
            patch.object(criterion, "get_bovw_dataframe", side_effect=self.frame),
            patch.object(VideoProcessor, "retrieve_segment_frames") as decode,
        ):
            criterion.evaluate(self.handler())
        requests = create.call_args.args[0]
        self.assertEqual(3, len(requests))
        self.assertEqual({("segment", 0, 5), ("segment", 5, 10)}, {value[1:] for value in requests.values()})
        self.assertEqual(3, workers.get.call_count)
        decode.assert_not_called()

    def test_video_mode_matches_legacy_dictionary_scores_and_rng_with_seed(self):
        rng = np.random.default_rng(793)
        frames = [rng.integers(0, 256, (96, 128), dtype=np.uint8) for _ in range(6)]
        handler = self.handler()
        group = handler.pick[0]
        features = {segment: ImageProcessor.ks_sift(frames) for segment in group}
        with threadpool_limits(limits=1):
            np.random.seed(734)
            legacy = BagOfVisualWords(features, dict_size=3)
            legacy.fit_kmeans()
            table = legacy.generate_bovw_dataframe()
            table["histogram_sum"] = table.sum(axis=1)
            chosen = set(table.nlargest(1, "histogram_sum").index)
            legacy_state = np.random.get_state()
            criterion = QualityPick("fixture", bovw_dict_size=3, visual_scope="video")
            single = SummarySegmentHandler()
            single.add_segments_to_pick(group, "fixture")
            np.random.seed(734)
            with patch.object(VideoProcessor, "retrieve_video_frames", return_value=frames):
                criterion.evaluate(single)
            current_state = np.random.get_state()
        self.assertEqual(chosen, single.include)
        np.testing.assert_array_equal(legacy_state[1], current_state[1])
        self.assertEqual(legacy_state[2:], current_state[2:])
        scores = {single.segment_id(segment): float(score) for segment, score in table["histogram_sum"].items()}
        self.assertEqual(scores, {candidate["segment_id"]: candidate["score"] for candidate in criterion.last_audit["clusters"][0]["candidates"]})


if __name__ == "__main__":
    unittest.main()
