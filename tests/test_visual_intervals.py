"""Native-frame interval bounds, checked seeking, and real decoder content."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from open_video_summary.utils.config import PROJECT_DIR
from open_video_summary.utils.processing.image import ImageProcessor
from open_video_summary.utils.processing.metrics import VisualProfile, collect_visual_profile
from open_video_summary.utils.processing.video import VideoProcessor


class CaptureFixture:
    def __init__(self, fps=29.97, count=120, *, seek=True, reported_offset=0, opened=True):
        self.fps = fps
        self.count = count
        self.seek = seek
        self.reported_offset = reported_offset
        self.opened = opened
        self.position = 0
        self.releases = 0
        self.read_indices = []
        self.seek_indices = []

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.count
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position + self.reported_offset
        raise AssertionError(prop)

    def set(self, prop, index):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.seek_indices.append(index)
        if self.seek:
            self.position = index
        return self.seek

    def read(self):
        if self.position >= self.count:
            return False, None
        index = self.position
        self.position += 1
        self.read_indices.append(index)
        return True, np.full((18, 32, 3), index, dtype=np.uint8)

    def release(self):
        self.releases += 1


class IntervalFrameTests(unittest.TestCase):
    def extract(self, capture, start, end, **kwargs):
        profile = VisualProfile()
        with (
            patch("open_video_summary.utils.processing.video.VideoCapture", return_value=capture),
            collect_visual_profile(profile),
        ):
            frames = VideoProcessor.retrieve_segment_frames("fixture.mp4", start, end, **kwargs)
        return frames, profile

    def test_nonzero_fractional_interval_keeps_global_stride_and_native_size(self):
        capture = CaptureFixture()
        frames, profile = self.extract(capture, 1.2, 3.0, grayscale=True)
        indices = [58, 87]
        self.assertEqual(indices, [int(frame[0, 0]) for frame in frames])
        self.assertTrue(all(frame.shape == (18, 32) for frame in frames))
        self.assertEqual([58], capture.seek_indices)
        self.assertEqual(58, capture.read_indices[0])
        self.assertEqual(1, capture.releases)
        self.assertEqual(32, profile.counters["decoded_frames"])
        self.assertEqual(2, profile.counters["sampled_frames"])
        self.assertEqual(1, profile.counters["interval_seeks"])

    def test_half_open_native_boundaries_and_fractional_fps(self):
        fps = 29.97
        cases = (
            (0, 58 / fps, [0, 29]),
            (29 / fps, 87 / fps, [29, 58]),
            (29 / fps + 1e-10, 87 / fps, [58]),
            (29 / fps - 1e-10, 87 / fps + 1e-10, [29, 58, 87]),
        )
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                frames, _ = self.extract(CaptureFixture(fps), start, end)
                self.assertEqual(expected, [int(frame[0, 0, 0]) for frame in frames])

    def test_empty_sampling_window_outside_source_and_end_past_source(self):
        for start, end in ((0.1, 0.2), (10, 11), (1e300, 1e308)):
            capture = CaptureFixture()
            frames, profile = self.extract(capture, start, end)
            self.assertEqual([], frames)
            self.assertEqual([], capture.read_indices)
            self.assertEqual(0, profile.counters["sampled_frames"])
            self.assertEqual(1, capture.releases)
        frames, _ = self.extract(CaptureFixture(), 2.5, 100)
        self.assertEqual([87, 116], [int(frame[0, 0, 0]) for frame in frames])

    def test_invalid_intervals_are_explicit_and_do_not_open_a_reader(self):
        for start, end in ((0, 0), (2, 1), (-1, 1), (0, float("nan")), (float("inf"), 3)):
            with self.subTest(start=start, end=end), patch(
                "open_video_summary.utils.processing.video.VideoCapture"
            ) as create, self.assertRaisesRegex(ValueError, "0 <= start < end"):
                VideoProcessor.retrieve_segment_frames("fixture.mp4", start, end)
            create.assert_not_called()

    def test_invalid_video_metadata_and_sampling_release_the_reader(self):
        cases = (
            CaptureFixture(fps=0), CaptureFixture(fps=float("nan")),
            CaptureFixture(count=0), CaptureFixture(opened=False),
        )
        for capture in cases:
            with self.subTest(fps=capture.fps, count=capture.count), self.assertRaisesRegex(ValueError, "FPS/frame count"):
                self.extract(capture, 0, 1)
            self.assertEqual(1, capture.releases)
        with self.assertRaisesRegex(ValueError, "sampling stride"):
            self.extract(CaptureFixture(fps=2), 0, 1, target_fps=3)

    def test_failed_or_unverified_seek_reopens_and_matches_sequential_content(self):
        for broken in (CaptureFixture(seek=False), CaptureFixture(reported_offset=1)):
            sequential = CaptureFixture()
            profile = VisualProfile()
            with (
                patch("open_video_summary.utils.processing.video.VideoCapture", side_effect=[broken, sequential]),
                collect_visual_profile(profile),
            ):
                frames = VideoProcessor.retrieve_segment_frames("fixture.mp4", 1.2, 3.0)
            self.assertEqual([58, 87], [int(frame[0, 0, 0]) for frame in frames])
            self.assertEqual(0, sequential.read_indices[0])
            self.assertEqual(1, broken.releases)
            self.assertEqual(1, sequential.releases)
            self.assertEqual(1, profile.counters["interval_seek_fallbacks"])

    def test_segment_sift_can_report_no_descriptors_without_changing_legacy_error(self):
        blank = [np.zeros((24, 24), dtype=np.uint8)] * 3
        for frames in ([], blank):
            actual = ImageProcessor.ks_sift(frames, allow_empty=True)
            self.assertEqual((0, 128), actual.shape)
            self.assertEqual(np.float32, actual.dtype)
            with self.assertRaises(ValueError):
                ImageProcessor.ks_sift(frames)


class RealDecoderIntervalTests(unittest.TestCase):
    def sequential_reference(self, path, start, end):
        video = cv2.VideoCapture(str(path))
        self.assertTrue(video.isOpened())
        fps = video.get(cv2.CAP_PROP_FPS)
        stride = int(fps)
        frames, indices = [], []
        index = 0
        try:
            while index / fps < end:
                success, frame = video.read()
                if not success:
                    break
                if index / fps >= start and index % stride == 0:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    indices.append(index)
                index += 1
        finally:
            video.release()
        return fps, indices, frames

    def compare_decoder(self, path, start, end):
        fps, indices, expected = self.sequential_reference(path, start, end)
        actual = VideoProcessor.retrieve_segment_frames(str(path), start, end, grayscale=True)
        self.assertGreater(len(actual), 0)
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
        return fps, indices

    def test_real_fractional_rate_seek_matches_sequential_native_frame_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fractional.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2.5, (64, 48))
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(111)
            for _ in range(40):
                writer.write(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
            writer.release()
            fps, indices = self.compare_decoder(path, 5.05, 11.2)
            self.assertEqual(2.5, fps)
            self.assertEqual(list(range(14, 28, 2)), indices)

    def test_local_mp4_gop_seek_matches_sequential_native_frame_pixels(self):
        path = PROJECT_DIR / "data/raw/catedral_notre_dame/jornal_nacional.mp4"
        if not path.is_file():
            self.skipTest("Local Cathedral source is not installed.")
        self.compare_decoder(path, 2.35, 6.91)


if __name__ == "__main__":
    unittest.main()
