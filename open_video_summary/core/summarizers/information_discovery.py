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
    timeout_fallback: object = None


class DiscoveryLookahead:
    """Bound each service independently while sharing a cap across its roles."""

    def __init__(self, concurrency, *, generator=None, evaluator=None):
        self.executor = ThreadPoolExecutor(max_workers=concurrency,
                                          thread_name_prefix="ovs-information-discovery")
        self.concurrency = concurrency
        self.lock = Lock()
        generator_key, evaluator_key = self._service_key(generator), self._service_key(evaluator)
        controller = getattr(generator, "controller", None)
        if controller is not None and controller is getattr(evaluator, "controller", None):
            generator_key = evaluator_key = ("shared_controller", id(controller))
        elif generator_key is None or evaluator_key is None:
            # Missing service identity never implies that two roles are independent.
            generator_key = evaluator_key = ("unknown_service",)
        self.service_keys = {"generator": generator_key, "evaluator": evaluator_key}
        self.slots = {key: BoundedSemaphore(concurrency) for key in set(self.service_keys.values())}
        self.stats = Counter()
        self.disabled = Counter()
        self.queue_seconds = 0.0
        self.request_wait_seconds = 0.0
        self.request_wait_by_role = Counter()

    @staticmethod
    def _service_key(adapter):
        config = getattr(adapter, "config", None)
        provider = getattr(config, "provider", None)
        if not isinstance(provider, str) or not provider.strip():
            return None
        endpoint = getattr(config, "base_url", "") or ""
        if not isinstance(endpoint, str):
            return None
        # Account, model and rate groups do not split the conservative concurrency cap.
        return (provider.strip().lower(), endpoint.rstrip("/"))

    @contextmanager
    def request_slot(self, signal, role):
        started = time.monotonic()
        slots = self.slots[self.service_keys[role]]
        while True:
            check_cancelled(signal)
            if slots.acquire(timeout=.05):
                break
        try:
            check_cancelled(signal)
            with self.lock:
                waited = time.monotonic() - started
                self.request_wait_seconds += waited
                self.request_wait_by_role[role] += waited
            yield
        finally:
            slots.release()

    def disable(self, reason):
        with self.lock:
            self.disabled[reason] += 1

    def submit(self, prompt, spec, reservation, callback, *, timeout_fallback=None):
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
        return DiscoveryFuture(prompt, spec, future, reservation, timeout_fallback)

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
                "request_slot_wait_seconds_by_role": {role: self.request_wait_by_role[role]
                                                       for role in self.service_keys},
                "service_scope_count": len(self.slots),
                "per_service_concurrency": self.concurrency,
                "concurrency_scope": "Provider and endpoint share a conservative cap across accounts, models and rate groups. Independent services have separate slots; unknown service identities share slots. Existing provider rate controllers remain unchanged.",
                "scheduling": "First independent QA/focus response lookahead after primary generation succeeds; canonical validation/application order and adaptive windows are unchanged.",
                "budget_guard": "Per-target worst-case canonical prefix headroom, including adaptive window attempts, candidate validation, literal repairs and joint checks. Actual lookahead requests reserve shared logical calls. Other concurrent targets can still consume the shared budget; this is not a global completeness guarantee."}
