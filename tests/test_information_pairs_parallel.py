"""Offline coverage for bounded semantic pair consolidation."""

import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_video_summary.adapters.typesafe import TypeSafeConfig, TypeSafeEvaluator

from open_video_summary.contracts import (
    ChoiceResult,
    EvaluationMetadata,
    EvaluationResult,
)
from open_video_summary.core.summarizers.information_analysis import (
    InformationAnalyzer,
    _AnalysisRun,
)
from open_video_summary.core.summarizers.information_config import (
    InformationAnalysisConfig,
)
from open_video_summary.core.summarizers.information_contracts import (
    AnalysisCall,
    CandidateRecord,
    Evidence,
    InformationCandidate,
    Qualifiers,
    capture_snapshot,
)
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.errors import AuthenticationError, ServiceTimeoutError


class OfflineGenerator:
    config = SimpleNamespace(provider="offline")
    records = []

    def preflight(self):
        return None


class PairTracker:
    def __init__(self, *, delays=None, failures=None, barrier=None, barrier_calls=3):
        self.delays = delays or {}
        self.failures = failures or {}
        self.barrier = barrier
        self.barrier_calls = barrier_calls
        self.cancel_on = None
        self.cancel_event = None
        self.held_pair = None
        self.release_held = threading.Event()
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.active = 0
        self.maximum_active = 0
        self.active_pairs = set()
        self.started = []
        self.completed = []

    @staticmethod
    def key(left, right):
        return frozenset((left, right))

    def evaluate(self, left, right):
        key = self.key(left, right)
        with self.changed:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.active_pairs.add(key)
            self.started.append((left, right))
            self.changed.notify_all()
            barrier = self.barrier if self.barrier_calls else None
            if barrier is not None:
                self.barrier_calls -= 1
        try:
            if key == self.held_pair and not self.release_held.wait(timeout=5):
                raise ServiceTimeoutError("held comparison timed out")
            if barrier is not None:
                barrier.wait(timeout=2)
            delay = self.delays.get(key, 0)
            if delay:
                time.sleep(delay)
            failure = self.failures.get(key)
            if failure is not None:
                raise failure
        finally:
            with self.changed:
                self.active -= 1
                self.active_pairs.discard(key)
                self.completed.append((left, right))
                self.changed.notify_all()
            if key == self.cancel_on and self.cancel_event is not None:
                self.cancel_event.set()

    def wait_for_starts(self, count, timeout):
        with self.changed:
            return self.changed.wait_for(lambda: len(self.started) >= count, timeout)


class ForkingEvaluator:
    can_fork = True

    def __init__(self, tracker=None, relations=None):
        self.tracker = tracker or PairTracker()
        self.relations = relations or {}
        self.config = SimpleNamespace(provider="offline")
        self.records = []

    def preflight(self):
        return None

    def fork(self):
        return ForkingEvaluator(self.tracker, self.relations)

    def evaluate(self, context, noul=None, choice=None):
        left = context.split("Left claim:\n", 1)[1].split("\n\n", 1)[0]
        right = context.split("Right claim:\n", 1)[1].split("\n\n", 1)[0]
        error = None
        try:
            self.tracker.evaluate(left, right)
        except Exception as exc:
            error = exc
        decisions = []
        for identifier, (_, options) in (choice or {}).items():
            selected = self.relations.get(
                frozenset((left, right)), "complementary"
            )
            confidence = 0.99
            probabilities = tuple(
                (
                    option,
                    0.99 if option == selected else 0.01 / max(1, len(options) - 1),
                )
                for option in options
            )
            decisions.append(
                ChoiceResult(identifier, selected, probabilities, confidence)
            )
        metadata = EvaluationMetadata(
            "offline",
            "offline",
            0.0,
            1,
            type(error).__name__ if error else "success",
            10,
            2,
            provider="offline",
            request_sent=True,
            wait_seconds=0.125,
        )
        self.records.append(metadata)
        if error is not None:
            raise error
        return EvaluationResult((), tuple(decisions), metadata)


def make_run(texts, *, config=None, tracker=None, relations=None, duplicates=()):
    content = ". ".join(texts) + "."
    source = [
        Video(
            "Fixture",
            "data/raw/fixture.mp4",
            topics=["Fixture"],
            segments=[VideoSegment(content, 0, 10, 0, "Fixture", "Fixture")],
        )
    ]
    snapshot = capture_snapshot(source)
    segment = snapshot.current_segments[0]
    records = []
    specifications = [(text, text) for text in texts]
    specifications.extend((text, text) for text in duplicates)
    for index, (text, quote) in enumerate(specifications):
        start = content.index(quote)
        evidence = Evidence(
            segment.id,
            segment.source_identity,
            quote,
            start,
            start + len(quote),
            "assertion",
            segment.start,
            segment.end,
        )
        candidate = InformationCandidate(
            text,
            "assertion",
            Qualifiers(None, False, None, (), ()),
            (evidence,),
            None,
            None,
            (),
        )
        records.append(
            CandidateRecord(
                f"c{index}",
                segment.id,
                "direct",
                0,
                candidate,
                "{}",
                "accepted",
                annotation_state="verified",
            )
        )

    evaluator = ForkingEvaluator(tracker, relations)
    analyzer = InformationAnalyzer(
        OfflineGenerator(),
        evaluator,
        config or InformationAnalysisConfig(pair_batch_size=1, qa_enabled=False),
    )
    run = _AnalysisRun(analyzer, snapshot)
    run.candidates.extend(records)
    run.evaluated_targets.update(snapshot.current_order)
    # Exercise consolidation and report metadata without invoking a provider.
    run._run_targets = lambda: None
    return run


class InformationPairParallelTests(unittest.TestCase):
    def test_pair_interrupt_cancels_native_retries_before_executor_shutdown(self):
        for origin in ("coordinator", "worker"):
            with self.subTest(origin=origin):
                run = make_run(["Backup every 24 hours", "Backup every 48 hours", "Backup every 72 hours"],
                    config=InformationAnalysisConfig(pair_batch_size=1, qa_enabled=False, pair_concurrency=2))
                both_started, release = threading.Event(), threading.Event()
                lock, sent = threading.Lock(), []
                def transport(url, **kwargs):
                    with lock:
                        sent.append(kwargs["body"])
                        ordinal = len(sent)
                        if ordinal == 2:
                            both_started.set()
                    if origin == "worker" and ordinal == 1:
                        if not both_started.wait(2):
                            raise AssertionError("Both first attempts must start")
                        raise KeyboardInterrupt()
                    waiting = release if origin == "coordinator" else run.budget.request_event
                    if not waiting.wait(2):
                        raise AssertionError("Interrupt did not release the in-flight request")
                    raise TimeoutError("offline")
                template = TypeSafeEvaluator(TypeSafeConfig(api_key="offline", max_attempts=2,
                    retry_backoff_seconds=0), transport=transport, jitter=lambda *_: 0)
                # Native worker objects share only their controller; transports
                # are injected at each worker boundary for this offline test.
                def fork():
                    return TypeSafeEvaluator(template.config, transport=transport,
                        jitter=lambda *_: 0, request_scope=template.request_scope)
                template.can_fork = True
                template.fork = fork
                run.evaluator = template
                if origin == "coordinator":
                    def interrupted_wait(*args, **kwargs):
                        if not both_started.wait(2):
                            raise AssertionError("Both first attempts must start")
                        release.set()
                        raise KeyboardInterrupt()
                    with patch("open_video_summary.core.summarizers.information_analysis.wait", interrupted_wait):
                        with self.assertRaises(KeyboardInterrupt):
                            run._consolidate()
                else:
                    with self.assertRaises(KeyboardInterrupt):
                        run._consolidate()
                self.assertTrue(run.budget.cancel_event.is_set())
                self.assertEqual(2, len(sent))
                self.assertEqual(2, run.budget.used)
                self.assertEqual(0, run.budget.reserved)

    def test_pair_interrupt_cancels_queued_futures_and_releases_unused_reservations(self):
        run = make_run(["Backup every 24 hours", "Backup every 48 hours", "Backup every 72 hours"],
            config=InformationAnalysisConfig(pair_batch_size=1, qa_enabled=False, pair_concurrency=2))
        started, queued_cancelled = threading.Event(), threading.Event()
        def held_worker(left, right, reservation, isolated):
            try:
                started.set()
                if not run.budget.request_event.wait(2):
                    raise AssertionError("Cancellation must precede executor shutdown")
                if not queued_cancelled.wait(2):
                    raise AssertionError("Queued work must be cancelled before shutdown")
            finally:
                reservation.release()
        with ThreadPoolExecutor(max_workers=1) as executor:
            original_submit, submitted = executor.submit, []
            def submit_worker(*args, **kwargs):
                future = original_submit(*args, **kwargs)
                submitted.append(future)
                if len(submitted) == 2:
                    future.add_done_callback(lambda item: queued_cancelled.set() if item.cancelled() else None)
                return future
            submit = Mock(side_effect=submit_worker)
            executor.submit = submit
            def interrupted_wait(*args, **kwargs):
                if not started.wait(2):
                    raise AssertionError("The first worker must start")
                raise KeyboardInterrupt()
            with patch.object(run, "_pair_worker", held_worker), patch(
                "open_video_summary.core.summarizers.information_analysis.ThreadPoolExecutor", return_value=executor), patch(
                "open_video_summary.core.summarizers.information_analysis.wait", interrupted_wait):
                with self.assertRaises(KeyboardInterrupt):
                    run._consolidate()
        self.assertEqual(2, submit.call_count)
        self.assertTrue(submitted[1].cancelled())
        self.assertTrue(run.budget.cancel_event.is_set())
        self.assertEqual(0, run.budget.used)
        self.assertEqual(0, run.budget.reserved)

    def test_comparisons_overlap_within_the_configured_bound(self):
        tracker = PairTracker(barrier=threading.Barrier(3))
        report = make_run(
            [
                "Backup every 24 hours",
                "Backup every 48 hours",
                "Backup every 72 hours",
                "Retention lasts 30 days",
                "Retention lasts 60 days",
            ],
            config=InformationAnalysisConfig(pair_batch_size=1,
                qa_enabled=False, pair_concurrency=3, max_pair_comparisons=20
            ),
            tracker=tracker,
        ).run()

        self.assertGreaterEqual(tracker.maximum_active, 2)
        self.assertLessEqual(tracker.maximum_active, 3)
        metadata = json.loads(report.metadata_json)
        self.assertEqual(10, metadata["paid_pair_comparisons"])
        self.assertEqual(10, metadata["physical_attempts"])
        self.assertEqual(0, metadata["retry_attempts"])
        self.assertEqual(1.25, metadata["wait_seconds"])

    def test_relations_calls_and_groups_are_stable_when_completion_order_changes(self):
        texts = [
            "Backup every 24 hours",
            "Backup every 48 hours",
            "Backup every 72 hours",
            "Retention lasts 30 days",
        ]
        relations = {
            frozenset((texts[0], texts[1])): "equivalent",
            frozenset((texts[0], texts[2])): "equivalent",
        }
        delayed = PairTracker(
            delays={
                frozenset((texts[0], texts[1])): 0.03,
                frozenset((texts[0], texts[2])): 0.01,
                frozenset((texts[1], texts[2])): 0.02,
            }
        )
        serial = make_run(
            texts,
            config=InformationAnalysisConfig(pair_batch_size=1,
                qa_enabled=False, pair_concurrency=1, max_pair_comparisons=20
            ),
            relations=relations,
        ).run()
        parallel = make_run(
            texts,
            config=InformationAnalysisConfig(pair_batch_size=1,
                qa_enabled=False, pair_concurrency=4, max_pair_comparisons=20
            ),
            relations=relations,
            tracker=delayed,
        ).run()

        self.assertEqual(
            [(item.left_candidate_id, item.right_candidate_id, item.relation)
             for item in serial.relations],
            [(item.left_candidate_id, item.right_candidate_id, item.relation)
             for item in parallel.relations],
        )
        self.assertEqual(
            [item.input_hash for item in serial.calls],
            [item.input_hash for item in parallel.calls],
        )
        self.assertEqual(
            [item.candidate_ids for item in serial.units],
            [item.candidate_ids for item in parallel.units],
        )

    def test_exact_pairs_are_free_even_when_paid_budget_and_cap_are_zero(self):
        run = make_run(
            ["Backup every 24 hours"],
            duplicates=("Backup every 24 hours",),
            config=InformationAnalysisConfig(pair_batch_size=1,
                qa_enabled=False, max_calls=1, max_pair_comparisons=0
            ),
        )
        run.budget.used = run.config.max_calls

        report = run.run()

        self.assertEqual(1, len(report.relations))
        self.assertEqual("exact_validated_proposition", report.relations[0].origin)
        metadata = json.loads(report.metadata_json)
        self.assertEqual(1, metadata["exact_pair_relations"])
        self.assertEqual(0, metadata["paid_pair_comparisons"])
        self.assertEqual(0, metadata["effective_pair_limit"])
        self.assertEqual(0, len(report.calls))

    def test_auto_limit_reserves_only_the_highest_priority_pairs(self):
        tracker = PairTracker()
        run = make_run(
            [
                "Backup every 24 hours",
                "Backup every 48 hours",
                "Backup every 72 hours",
                "Retention lasts 30 days",
            ],
            config=InformationAnalysisConfig(pair_batch_size=1,
                qa_enabled=False, max_calls=5, max_pair_comparisons=None,
                pair_concurrency=8,
            ),
            tracker=tracker,
        )
        run.budget.used = 3

        report = run.run()

        metadata = json.loads(report.metadata_json)
        self.assertEqual(2, metadata["effective_pair_limit"])
        self.assertEqual(2, metadata["paid_pair_comparisons"])
        self.assertEqual(
            [("c0", "c1"), ("c0", "c2")],
            [(item.left_candidate_id, item.right_candidate_id)
             for item in report.relations],
        )
        self.assertEqual(2, len(tracker.started))
        self.assertEqual(4, metadata["unexamined_pair_count"])

    def test_terminal_provider_failure_prevents_later_pair_submissions(self):
        texts = [
            "Backup every 24 hours",
            "Backup every 48 hours",
            "Backup every 72 hours",
        ]
        first = frozenset((texts[0], texts[1]))
        tracker = PairTracker(failures={first: AuthenticationError("denied")})
        report = make_run(
            texts,
            config=InformationAnalysisConfig(pair_batch_size=1,
                qa_enabled=False, pair_concurrency=1, max_pair_comparisons=20
            ),
            tracker=tracker,
        ).run()

        self.assertEqual(1, len(tracker.started))
        metadata = json.loads(report.metadata_json)
        self.assertEqual(1, metadata["paid_pair_comparisons"])
        self.assertEqual("AuthenticationError", metadata["permanent_provider_failure"])
        self.assertFalse(report.relations)

    def test_timeout_keeps_valid_decisions_from_the_same_inflight_batch(self):
        texts = [
            "Backup every 24 hours",
            "Backup every 48 hours",
            "Backup every 72 hours",
        ]
        failed = frozenset((texts[0], texts[1]))
        valid = frozenset((texts[0], texts[2]))
        tracker = PairTracker(failures={failed: ServiceTimeoutError("timeout")})
        report = make_run(
            texts,
            config=InformationAnalysisConfig(pair_batch_size=1,
                qa_enabled=False, pair_concurrency=2, max_pair_comparisons=20
            ),
            tracker=tracker,
        ).run()

        self.assertEqual(3, len(tracker.started))
        self.assertEqual(
            [("c0", "c2"), ("c1", "c2")],
            [(item.left_candidate_id, item.right_candidate_id)
             for item in report.relations],
        )
        self.assertIn("pair_evaluation_failed", {item.kind for item in report.issues})
        audit = json.loads(report.metadata_json)
        self.assertEqual(3, audit["physical_attempts"])
        self.assertEqual({"ServiceTimeoutError": 1}, audit["provider_audit"]["offline"]["errors"])

    def test_nonforkable_evaluator_keeps_the_serial_fallback(self):
        tracker = PairTracker()
        run = make_run(
            ["Backup every 24 hours", "Backup every 48 hours", "Retention lasts 30 days"],
            config=InformationAnalysisConfig(pair_batch_size=1,
                qa_enabled=False, pair_concurrency=8, max_pair_comparisons=20
            ),
            tracker=tracker,
        )
        run.evaluator.can_fork = False

        report = run.run()

        self.assertEqual(1, json.loads(report.metadata_json)["pair_concurrency"])
        self.assertEqual(1, tracker.maximum_active)

    def test_cancellation_stops_before_the_next_pair_batch(self):
        texts = [
            "Backup every 24 hours",
            "Backup every 48 hours",
            "Backup every 72 hours",
        ]
        tracker = PairTracker()
        tracker.cancel_on = frozenset((texts[0], texts[1]))
        run = make_run(
            texts,
            config=InformationAnalysisConfig(pair_batch_size=1,
                qa_enabled=False, pair_concurrency=1, max_pair_comparisons=20
            ),
            tracker=tracker,
        )
        tracker.cancel_event = run.budget.cancel_event

        report = run.run()

        self.assertEqual(1, len(tracker.started))
        self.assertEqual(1, json.loads(report.metadata_json)["paid_pair_comparisons"])
        self.assertEqual(1, len(report.relations))

    def test_provider_audit_counts_only_attempts_marked_as_sent(self):
        call = AnalysisCall(
            "compare_candidates",
            "offline",
            "input",
            None,
            "ServiceTimeoutError",
            json.dumps(
                {
                    "attempt_records": [
                        {
                            "attempts": 1,
                            "request_sent": False,
                            "wait_seconds": 0.5,
                            "status": "RequestDeadlineError",
                        },
                        {
                            "attempts": 2,
                            "request_sent": True,
                            "wait_seconds": 0.1,
                            "status": "ServiceTimeoutError",
                        },
                        {
                            "attempts": 3,
                            "request_sent": True,
                            "wait_seconds": 0.0,
                            "status": "success",
                        },
                    ]
                }
            ),
        )

        audit = _AnalysisRun._provider_attempt_audit((call,))

        self.assertEqual(2, audit["physical_attempts"])
        self.assertEqual(1, audit["retry_attempts"])
        self.assertEqual(0.6, audit["wait_seconds"])
        self.assertEqual(
            {"RequestDeadlineError": 1, "ServiceTimeoutError": 1},
            audit["provider_audit"]["offline"]["errors"],
        )

    def test_rolling_scheduler_starts_later_pairs_while_the_first_is_blocked(self):
        tracker = PairTracker()
        texts = [
            "Backup every 24 hours",
            "Backup every 48 hours",
            "Backup every 72 hours",
            "Retention lasts 30 days",
            "Retention lasts 60 days",
        ]
        run = make_run(
            texts,
            config=InformationAnalysisConfig(pair_batch_size=1,
                qa_enabled=False, pair_concurrency=8, max_pair_comparisons=20
            ),
            tracker=tracker,
        )
        first_pair = run._candidate_pairs(run._accepted(), 20)[1][0]
        tracker.held_pair = frozenset(
            (first_pair[0].candidate.text, first_pair[1].candidate.text)
        )
        result = {}
        failures = []

        def analyze():
            try:
                result["report"] = run.run()
            except BaseException as exc:
                failures.append(exc)

        thread = threading.Thread(target=analyze)
        thread.start()
        try:
            self.assertTrue(tracker.wait_for_starts(9, timeout=2))
            self.assertIn(tracker.held_pair, tracker.active_pairs)
        finally:
            tracker.release_held.set()
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        self.assertEqual(10, len(result["report"].relations))


if __name__ == "__main__":
    unittest.main()
