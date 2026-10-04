"""Bounded process extraction, ordered consumption, and Windows spawn coverage."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import numpy as np

from open_video_summary.utils.config import PROJECT_DIR
from open_video_summary.utils.processing.metrics import (
    VisualProfile,
    collect_visual_profile,
)
from open_video_summary.utils.processing.parallel import (
    DescriptorWorkers,
    source_signature,
)


class DescriptorWorkerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.sources = {}
        for index in range(5):
            path = self.directory / f"source_{index}.mp4"
            path.write_bytes(bytes([index]))
            self.sources[source_signature(str(path))] = str(path)
        self.executor = Mock()
        self.executor.submit.side_effect = self.submit
        self.failure_task = None
        self.profile = VisualProfile()

    def submit(self, function, path, signature, output, task):
        future = Future()
        if task == self.failure_task:
            future.set_exception(ValueError("fixture extraction failed"))
        else:
            np.save(output, np.full((2, 128), task, dtype=np.float32))
            future.set_result(
                {
                    "stale": False,
                    "output": output,
                    "worker": {
                        "pid": task + 100,
                        "cpu_seconds": 2.0,
                        "stage_seconds": {"sift_detect": 1.0},
                    },
                    "counters": {"video_reads": 1},
                }
            )
        return future

    def test_window_is_bounded_and_results_keep_request_order(self):
        with (
            patch(
                "open_video_summary.utils.processing.parallel.ProcessPoolExecutor",
                return_value=self.executor,
            ) as create,
            collect_visual_profile(self.profile),
        ):
            with DescriptorWorkers(self.sources, workers=2) as workers:
                temporary = Path(workers.directory.name)
                self.assertEqual(3, self.executor.submit.call_count)
                self.assertEqual(3, workers.pending_limit)
                for index, key in enumerate(self.sources):
                    actual = workers.get(key)
                    np.testing.assert_array_equal(
                        np.full((2, 128), index, dtype=np.float32), actual
                    )
                    self.assertLessEqual(len(workers.pending), 3)
                    self.assertLessEqual(len(list(temporary.glob("*.npy"))), 3)
                    submitted = self.executor.submit.call_count
                    workers.wait_before_fit()
                    self.assertEqual(submitted, self.executor.submit.call_count)
            self.assertFalse(temporary.exists())
        self.assertEqual(
            "spawn", create.call_args.kwargs["mp_context"].get_start_method()
        )
        self.assertEqual(2, create.call_args.kwargs["max_workers"])
        self.executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        report = self.profile.as_dict()
        self.assertEqual(3, report["counters"]["worker_pending_limit"])
        self.assertEqual(3, report["counters"]["worker_pending_peak"])
        self.assertEqual(5, report["counters"]["video_reads"])
        self.assertEqual(10.0, report["worker_cpu_seconds"])
        self.assertEqual({"sift_detect": 5.0}, report["worker_stage_seconds"])
        self.assertNotIn("sift_detect", report["stage_seconds"])

    def test_third_task_starts_before_first_finishes_when_second_completes(self):
        release_first = Event()
        second_finished = Event()
        third_started = Event()

        def extract(path, signature, output, task):
            if task == 0:
                self.assertTrue(release_first.wait(timeout=15))
            elif task == 2:
                self.assertTrue(second_finished.is_set())
                third_started.set()
            result = self.submit(None, path, signature, output, task).result()
            if task == 1:
                second_finished.set()
            return result

        # Use real executor scheduling with two threads, without numerical work
        # or process startup. The production executor still uses Windows spawn.
        with ThreadPoolExecutor(max_workers=2) as executor:
            with (
                patch(
                    "open_video_summary.utils.processing.parallel.ProcessPoolExecutor",
                    return_value=executor,
                ) as create,
                patch(
                    "open_video_summary.utils.processing.parallel._extract_to_file",
                    side_effect=extract,
                ),
                collect_visual_profile(self.profile),
                DescriptorWorkers(self.sources, workers=2) as workers,
            ):
                temporary = Path(workers.directory.name)
                try:
                    self.assertTrue(second_finished.wait(timeout=5))
                    self.assertTrue(third_started.wait(timeout=5))
                    keys = list(self.sources)
                    self.assertFalse(workers.pending[keys[0]].done())
                    self.assertEqual(3, len(workers.pending))
                    self.assertEqual(2, create.call_args.kwargs["max_workers"])
                finally:
                    release_first.set()
                for task, key in enumerate(keys):
                    np.testing.assert_array_equal(
                        np.full((2, 128), task, dtype=np.float32), workers.get(key)
                    )
                    self.assertLessEqual(len(workers.pending), 3)
        self.assertFalse(temporary.exists())
        self.assertEqual(3, self.profile.counters["worker_pending_peak"])

    def test_pending_limit_is_capped_by_unique_source_count(self):
        with (
            patch(
                "open_video_summary.utils.processing.parallel.ProcessPoolExecutor",
                return_value=self.executor,
            ) as create,
            collect_visual_profile(self.profile),
            DescriptorWorkers(self.sources, workers=8) as workers,
        ):
            self.assertEqual(5, create.call_args.kwargs["max_workers"])
            self.assertEqual(5, workers.pending_limit)
            self.assertEqual(5, self.executor.submit.call_count)
            self.assertEqual(5, len(workers.pending))
        self.assertEqual(5, self.profile.counters["worker_pending_limit"])
        self.assertEqual(5, self.profile.counters["worker_pending_peak"])

    def test_changed_source_closes_speculation_before_serial_fallback(self):
        with (
            patch(
                "open_video_summary.utils.processing.parallel.ProcessPoolExecutor",
                return_value=self.executor,
            ),
            collect_visual_profile(self.profile),
            DescriptorWorkers(self.sources, workers=2) as workers,
        ):
            temporary = Path(workers.directory.name)
            key = next(iter(self.sources))
            Path(self.sources[key]).write_bytes(b"modified")
            self.assertIsNone(workers.get(key))
            self.assertTrue(workers.closed)
            self.assertFalse(temporary.exists())
        self.assertEqual(1, self.profile.counters["worker_stale_results"])
        self.assertEqual(3, self.profile.counters["video_reads"])

    def test_evicted_request_falls_back_without_growing_pending_window(self):
        with (
            patch(
                "open_video_summary.utils.processing.parallel.ProcessPoolExecutor",
                return_value=self.executor,
            ),
            collect_visual_profile(self.profile),
            DescriptorWorkers(self.sources, workers=2) as workers,
        ):
            key = next(iter(self.sources))
            workers.get(key)
            self.assertIsNone(workers.get(key))
            self.assertTrue(workers.closed)
            self.assertEqual(4, self.executor.submit.call_count)
        self.assertEqual(1, self.profile.counters["worker_fallbacks"])
        self.assertEqual(4, self.profile.counters["video_reads"])

    def test_later_failure_is_raised_in_request_order_and_cleans_everything(self):
        self.failure_task = 1
        with self.assertRaisesRegex(ValueError, "fixture extraction failed"):
            with (
                patch(
                    "open_video_summary.utils.processing.parallel.ProcessPoolExecutor",
                    return_value=self.executor,
                ),
                collect_visual_profile(self.profile),
                DescriptorWorkers(self.sources, workers=2) as workers,
            ):
                temporary = Path(workers.directory.name)
                keys = list(self.sources)
                workers.get(keys[0])
                workers.wait_before_fit()
                self.assertFalse(workers.closed)
                workers.get(keys[1])
        self.assertFalse(temporary.exists())
        self.assertEqual("failed", self.profile.status)
        self.assertEqual(1, self.profile.counters["worker_failed_tasks"])
        self.executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


class RealSpawnTests(unittest.TestCase):
    def test_spawn_from_foreign_cwd_preserves_exact_features_rng_and_selections(self):
        # A separate guarded main file exercises Windows spawn without inheriting
        # unittest mocks. It deliberately imports NumPy before worker initialization.
        script = textwrap.dedent(
            r"""
            import json
            import os
            import random
            import sys
            from pathlib import Path
            import cv2
            import numpy as np
            from threadpoolctl import threadpool_limits
            from open_video_summary.core.selection_criteria.quality import QualityPick
            from open_video_summary.entities.video import VideoSegment
            from open_video_summary.handlers.summary import SummarySegmentHandler

            def main():
                directory = Path(sys.argv[1])
                rng = np.random.default_rng(713)
                groups = []
                sources = []
                for source in range(3):
                    path = directory / f"source_{source}.avi"
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1, (128, 96))
                    assert writer.isOpened()
                    for _ in range(12):
                        writer.write(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8))
                    writer.release()
                    sources.append(path)
                for group in range(3):
                    start = (group % 2) * 4
                    groups.append({VideoSegment(f"item{source}/{group}", start, start + 5,
                        order=group, video_path=str(path)) for source, path in enumerate(sources)})

                profiles = {}
                cv2.setNumThreads(1)
                with threadpool_limits(limits=1):
                    for scope in ("video", "segment"):
                        outputs = []
                        for budget in (1, 2):
                            handler = SummarySegmentHandler()
                            for group in groups:
                                handler.add_segments_to_pick(group, "fixture")
                            criterion = QualityPick("fixture", bovw_dict_size=3,
                                visual_threads=budget, visual_scope=scope)
                            extract = criterion.extract_segments_visual_features
                            records = []
                            def capture(segments):
                                result = extract(segments)
                                records.extend((segment, array.copy()) for segment, array in result.items())
                                return result
                            criterion.extract_segments_visual_features = capture
                            random.seed(7319)
                            np.random.seed(7319)
                            criterion.evaluate(handler)
                            outputs.append((handler, records, np.random.get_state(),
                                criterion.last_profile, criterion.last_audit))
                        serial, parallel = outputs
                        assert serial[0].include == parallel[0].include
                        assert serial[4] == parallel[4]
                        assert len(serial[1]) == len(parallel[1]) == 9
                        for (left_segment, left), (right_segment, right) in zip(serial[1], parallel[1]):
                            assert left_segment == right_segment
                            np.testing.assert_array_equal(left, right)
                        assert serial[2][0] == parallel[2][0]
                        np.testing.assert_array_equal(serial[2][1], parallel[2][1])
                        assert serial[2][2:] == parallel[2][2:]
                        if scope == "segment":
                            arrays = {(segment.video_path, segment.order): array for segment, array in serial[1]}
                            for path in sources:
                                assert not np.array_equal(arrays[(str(path), 0)], arrays[(str(path), 1)])
                                np.testing.assert_array_equal(arrays[(str(path), 0)], arrays[(str(path), 2)])
                        profile = parallel[3]
                        unique = 6 if scope == "segment" else 3
                        assert profile["counters"]["video_reads"] == unique
                        assert profile["counters"]["cache_hits"] == 9 - unique
                        assert profile["counters"]["parallel_workers"] == 2
                        assert profile["counters"]["worker_pending_limit"] == 3
                        assert profile["counters"]["worker_pending_peak"] == 3
                        assert len(profile["workers"]) == unique
                        assert len({worker["pid"] for worker in profile["workers"]}) <= 2
                        assert profile["settings"]["scope"] == scope
                        for worker in profile["workers"]:
                            assert worker["pid"] != os.getpid()
                            assert worker["opencv_threads"] == 1
                            assert all(pool["num_threads"] == 1 for pool in worker["native_pools"])
                        profiles[scope] = profile
                Path(sys.argv[2]).write_text(json.dumps(profiles), encoding="utf-8")

            if __name__ == "__main__":
                main()
            """
        )
        output = PROJECT_DIR / "outputs"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as directory:
            directory = Path(directory)
            program = directory / "spawn_check.py"
            report = directory / "report.json"
            program.write_text(script, encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(PROJECT_DIR)
            result = subprocess.run(
                [sys.executable, str(program), str(directory), str(report)],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            saved = json.loads(report.read_text())
            self.assertEqual("completed", saved["video"]["status"])
            self.assertEqual("completed", saved["segment"]["status"])


if __name__ == "__main__":
    unittest.main()
