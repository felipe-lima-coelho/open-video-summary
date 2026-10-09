"""Bounded lookahead for immutable first discovery requests.

Only responses are fetched ahead. The target coordinator still applies them,
validates candidates and advances adaptive windows in its original order.
"""

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock

from open_video_summary.utils.retry import check_cancelled


@dataclass(frozen=True)
class DiscoveryOutcome:
    result: object
    calls: tuple
    issues: tuple
    error: BaseException | None = None


@dataclass(frozen=True)
class DiscoveryFuture:
    prompt: str
    spec: object
    future: object
    reservation: object


class DiscoveryLookahead:
    """Share one physical-request bound with all target generation/evaluation."""

    def __init__(self, concurrency):
        self.executor = ThreadPoolExecutor(max_workers=concurrency,
                                          thread_name_prefix="ovs-information-discovery")
        self.slots = BoundedSemaphore(concurrency)
        self.lock = Lock()
        self.stats = Counter()
        self.disabled = Counter()
        self.queue_seconds = 0.0
        self.request_wait_seconds = 0.0

    @contextmanager
    def request_slot(self, signal):
        started = time.monotonic()
        while True:
            check_cancelled(signal)
            if self.slots.acquire(timeout=.05):
                break
        try:
            check_cancelled(signal)
            with self.lock:
                self.request_wait_seconds += time.monotonic() - started
            yield
        finally:
            self.slots.release()

    def disable(self, reason):
        with self.lock:
            self.disabled[reason] += 1

    def submit(self, prompt, spec, reservation, callback):
        queued = time.monotonic()
        with self.lock:
            self.stats["eligible"] += 1
            self.stats["submitted"] += 1

        def execute():
            with self.lock:
                self.stats["started"] += 1
                self.queue_seconds += time.monotonic() - queued
            try:
                return callback()
            finally:
                reservation.release()

        try:
            future = self.executor.submit(execute)
        except BaseException:
            reservation.release()
            raise
        return DiscoveryFuture(prompt, spec, future, reservation)

    def consumed(self):
        with self.lock:
            self.stats["consumed"] += 1

    def discarded(self, item):
        cancelled = item.future.cancel()
        if cancelled:
            item.reservation.release()
        with self.lock:
            self.stats["discarded"] += 1
            self.stats["cancelled_before_start"] += int(cancelled)
        return None if cancelled else item.future.result()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def snapshot(self):
        with self.lock:
            return {"enabled": True, **{key: self.stats[key] for key in (
                "eligible", "submitted", "started", "consumed", "discarded",
                "cancelled_before_start")},
                "disabled_reasons": dict(self.disabled),
                "queue_seconds": self.queue_seconds,
                "request_slot_wait_seconds": self.request_wait_seconds,
                "scheduling": "First independent QA/focus response lookahead after primary generation succeeds; canonical validation/application order and adaptive windows are unchanged.",
                "budget_guard": "Per-target worst-case canonical prefix headroom, including adaptive window attempts, candidate validation, literal repairs and joint checks. Actual lookahead requests reserve shared logical calls. Other concurrent targets can still consume the shared budget; this is not a global completeness guarantee."}
