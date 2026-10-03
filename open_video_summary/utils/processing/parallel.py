"""Bounded, per-evaluation video extraction with Windows spawn workers."""

import os
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

from open_video_summary.utils.processing.metrics import (
    visual_count,
    visual_maximum,
    visual_stage,
    visual_worker,
)


def source_signature(path: str) -> tuple:
    """Use the same file identity fields as the descriptor cache."""
    stat = Path(path).stat()
    return (
        os.path.normcase(str(Path(path).resolve())),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _initialize_worker() -> None:
    # Set these before the worker's numerical imports. The runtime limiter below
    # also covers libraries imported when spawn replays the caller's main module.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _extract_to_file(path: str, expected_signature: tuple, output: str, task: int):
    imports_started = perf_counter()
    imports_cpu_started = process_time()
    import cv2
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits

    from open_video_summary.utils.processing.image import ImageProcessor
    from open_video_summary.utils.processing.metrics import (
        VisualProfile,
        collect_visual_profile,
    )
    from open_video_summary.utils.processing.video import VideoProcessor

    imports_seconds = perf_counter() - imports_started
    imports_cpu_seconds = process_time() - imports_cpu_started
    cv2.setNumThreads(1)
    if source_signature(path) != expected_signature:
        return {"stale": True}

    profile = VisualProfile()
    with threadpool_limits(limits=1):
        native_pools = [
            {key: pool[key] for key in ("user_api", "internal_api", "num_threads")}
            for pool in threadpool_info()
        ]
        started = perf_counter()
        cpu_started = process_time()
        with collect_visual_profile(profile):
            frames = VideoProcessor.retrieve_video_frames(path, grayscale=True)
            try:
                descriptors = ImageProcessor.ks_sift(frames)
            finally:
                del frames
        finished = perf_counter()
        cpu_seconds = process_time() - cpu_started

    worker = {
        "task": task,
        "pid": os.getpid(),
        "started_seconds": started,
        "finished_seconds": finished,
        "wall_seconds": finished - started,
        "cpu_seconds": cpu_seconds,
        "imports_seconds": imports_seconds,
        "imports_cpu_seconds": imports_cpu_seconds,
        "opencv_threads": cv2.getNumThreads(),
        "native_pools": native_pools,
        "stage_seconds": profile.as_dict()["stage_seconds"],
    }
    result = {
        "stale": source_signature(path) != expected_signature,
        "worker": worker,
        "counters": profile.as_dict()["counters"],
    }
    if not result["stale"]:
        # Completed futures retain only metadata. Descriptor arrays waiting for
        # their original request order live on disk, outside the in-memory LRU.
        started = perf_counter()
        np.save(output, descriptors, allow_pickle=False)
        worker["spool_write_seconds"] = perf_counter() - started
        result["output"] = output
    return result


class DescriptorWorkers:
    """Prefetch at most ``workers + 1`` videos and consume in request order."""

    def __init__(self, sources: dict, workers: int) -> None:
        self.workers = min(workers, len(sources))
        self.pending_limit = min(self.workers + 1, len(sources))
        self.sources = iter(sources.items())
        self.pending = OrderedDict()
        self.executor = None
        self.directory = None
        self.closed = False
        self.next_task = 0

    def __enter__(self):
        try:
            self.directory = TemporaryDirectory(prefix="ovs-visual-")
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_worker,
            )
            visual_maximum("parallel_workers", self.workers)
            visual_maximum("worker_pending_limit", self.pending_limit)
            self._fill()
            return self
        except BaseException:
            self.close()
            raise

    def _fill(self) -> None:
        # One queued task can keep a worker busy when a later video finishes
        # before the first requested result. Pending arrays stay on disk.
        while not self.closed and len(self.pending) < self.pending_limit:
            source = next(self.sources, None)
            if source is None:
                break
            key, path = source
            task = self.next_task
            self.next_task += 1
            output = str(Path(self.directory.name) / f"{task}.npy")
            self.pending[key] = self.executor.submit(
                _extract_to_file, path, key[:6], output, task
            )
            visual_count("worker_tasks_submitted")
            visual_maximum("worker_pending_peak", len(self.pending))

    def get(self, key):
        if self.closed:
            return None
        if key not in self.pending:
            # An evicted/oversized entry or changed file needs another read.
            # Finish the bounded speculative window, then resume the original
            # serial path; never grow an additional queue of pending arrays.
            visual_count("worker_fallbacks")
            self.close()
            return None

        with visual_stage("worker_wait"):
            result = self.pending[key].result()
        del self.pending[key]
        if "worker" in result:
            visual_worker(result["worker"], result["counters"])
        if result["stale"] or source_signature(key[0]) != key[:6]:
            visual_count("worker_stale_results")
            self.close()
            return None

        import numpy as np

        with visual_stage("descriptor_spool_read"):
            try:
                descriptors = np.load(result["output"], allow_pickle=False)
            finally:
                Path(result["output"]).unlink(missing_ok=True)
        self._fill()
        return descriptors

    def wait_before_fit(self) -> None:
        if not self.closed and self.pending:
            # A parent KMeans fit can itself use the CLI thread budget. Do not
            # overlap it with the extraction workers. Waiting does not refill
            # the window or surface a later video's exception ahead of its turn.
            with visual_stage("worker_wait"):
                wait(self.pending.values())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.executor is not None:
                with visual_stage("worker_shutdown"):
                    self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            try:
                for future in self.pending.values():
                    if future.cancelled() or not future.done():
                        continue
                    try:
                        result = future.result()
                    except BaseException:
                        visual_count("worker_failed_tasks")
                    else:
                        if "worker" in result:
                            visual_worker(result["worker"], result["counters"])
            finally:
                self.pending.clear()
                if self.directory is not None:
                    self.directory.cleanup()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
