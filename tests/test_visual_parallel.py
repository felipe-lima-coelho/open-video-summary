"""Bounded process extraction, ordered consumption, and Windows spawn coverage."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from concurrent.futures import Future
from pathlib import Path
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
                self.assertEqual(2, self.executor.submit.call_count)
                for index, key in enumerate(self.sources):
                    actual = workers.get(key)
                    np.testing.assert_array_equal(
                        np.full((2, 128), index, dtype=np.float32), actual
                    )
                    self.assertLessEqual(len(workers.pending), 2)
                    self.assertLessEqual(len(list(temporary.glob("*.npy"))), 2)
                    submitted = self.executor.submit.call_count
                    workers.wait_before_fit()
                    self.assertEqual(submitted, self.executor.submit.call_count)
            self.assertFalse(temporary.exists())
        self.assertEqual(
            "spawn", create.call_args.kwargs["mp_context"].get_start_method()
        )
        self.executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        report = self.profile.as_dict()
        self.assertEqual(5, report["counters"]["video_reads"])
        self.assertEqual(10.0, report["worker_cpu_seconds"])
        self.assertEqual({"sift_detect": 5.0}, report["worker_stage_seconds"])
        self.assertNotIn("sift_detect", report["stage_seconds"])

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
        self.assertEqual(2, self.profile.counters["video_reads"])

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
            self.assertEqual(3, self.executor.submit.call_count)
        self.assertEqual(1, self.profile.counters["worker_fallbacks"])
        self.assertEqual(3, self.profile.counters["video_reads"])

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
                for source in range(2):
                    path = directory / f"source_{source}.avi"
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1, (128, 96))
                    assert writer.isOpened()
                    for _ in range(5):
                        writer.write(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8))
                    writer.release()
                    sources.append(path)
                for group in range(2):
                    groups.append({VideoSegment(f"item{source}/{group}", group, group + 1,
                        order=group, video_path=str(path)) for source, path in enumerate(sources)})

                outputs = []
                cv2.setNumThreads(1)
                with threadpool_limits(limits=1):
                    for budget in (1, 2):
                        handler = SummarySegmentHandler()
                        for group in groups:
                            handler.add_segments_to_pick(group, "fixture")
                        criterion = QualityPick("fixture", bovw_dict_size=3, visual_threads=budget)
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
                        outputs.append((handler, records, np.random.get_state(), criterion.last_profile))
                serial, parallel = outputs
                assert serial[0].include == parallel[0].include
                assert len(serial[1]) == len(parallel[1]) == 4
                for (left_segment, left), (right_segment, right) in zip(serial[1], parallel[1]):
                    assert left_segment == right_segment
                    np.testing.assert_array_equal(left, right)
                assert serial[2][0] == parallel[2][0]
                np.testing.assert_array_equal(serial[2][1], parallel[2][1])
                assert serial[2][2:] == parallel[2][2:]
                profile = parallel[3]
                assert profile["counters"]["video_reads"] == 2
                assert profile["counters"]["cache_hits"] == 2
                assert profile["counters"]["parallel_workers"] == 2
                assert len(profile["workers"]) == 2
                for worker in profile["workers"]:
                    assert worker["pid"] != os.getpid()
                    assert worker["opencv_threads"] == 1
                    assert all(pool["num_threads"] == 1 for pool in worker["native_pools"])
                Path(sys.argv[2]).write_text(json.dumps(profile), encoding="utf-8")

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
            self.assertEqual("completed", json.loads(report.read_text())["status"])


if __name__ == "__main__":
    unittest.main()
