"""Offline concurrency, request ownership, bounded retries and progress tests."""

import contextlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_video_summary.adapters.llm import OpenAIAdapter
from open_video_summary import __main__ as cli
from open_video_summary.adapters.typesafe import TransportResponse, TypeSafeConfig, TypeSafeEvaluator
from open_video_summary.contracts import GenerationRequest, ProviderProgress
from open_video_summary.core.summarizers.base import Summarizer
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig, configured_information_analyzer
from open_video_summary.core.summarizers.information_contracts import AnalysisProgress, capture_snapshot
from open_video_summary.errors import AuthenticationError, ConfigurationError, ProviderConfigurationError, ServiceTimeoutError
from open_video_summary.utils.providers import LLMConfig
from open_video_summary.utils.retry import retry_after, retry_delay
from tests.test_information_analysis import ScriptedGenerator, SyntheticEvaluator, candidate, videos
from tests.test_llm_adapters import ExternalStatusError, response


class _Requests:
    def __init__(self, *, synchronize=False, delays=None, failures=None, transient=False,
                 evaluation_failures=None):
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(2) if synchronize else None
        self.delays, self.failures = delays or {}, failures or {}
        self.transient = transient
        self.evaluation_failures = evaluation_failures or {}
        self.active = self.maximum = 0
        self.targets, self.evaluation_ids, self.clients = [], [], []
        self.target_attempts = {}

    def client(self, **kwargs):
        tracker = self

        class Client:
            def __init__(self):
                self.settings, self.closed = kwargs, False
                self.responses = SimpleNamespace(create=self.create)

            def create(self, **request):
                data = json.loads(request["input"].split("\nInput:\n", 1)[1])
                identifier = data["target"]["id"]
                with tracker.lock:
                    tracker.active += 1
                    tracker.maximum = max(tracker.maximum, tracker.active)
                    tracker.targets.append(identifier)
                    attempt = tracker.target_attempts.get(identifier, 0) + 1
                    tracker.target_attempts[identifier] = attempt
                try:
                    if tracker.barrier and len(tracker.targets) <= 2:
                        tracker.barrier.wait(timeout=3)
                    time.sleep(tracker.delays.get(identifier, 0.005))
                    if identifier in tracker.failures:
                        raise tracker.failures[identifier]
                    if tracker.transient and attempt == 1:
                        raise TimeoutError("private transcript and credential")
                    value = {"candidates": [candidate(identifier, data["target"]["text"])], "issues": []}
                    return response(json.dumps(value, ensure_ascii=False))
                finally:
                    with tracker.lock:
                        tracker.active -= 1

            def close(self):
                self.closed = True

        client = Client()
        self.clients.append(client)
        return client

    def evaluate(self, url, *, headers, body, timeout):
        payload = json.loads(body)
        state = json.loads(payload["state"])
        with self.lock:
            if "candidate_id" in state:
                self.evaluation_ids.append((state["target_id"], state["candidate_id"]))
        if state.get("target_id") in self.evaluation_failures:
            return TransportResponse(self.evaluation_failures[state["target_id"]], {}, b"private service detail")
        answers = {}
        for identifier, question in payload["questions"].items():
            if question["type"] == "noul":
                answers[identifier] = {"type": "noul", "noul": 0.01 if "original_target" in state else 0.99}
            else:
                options = tuple(question["criteria"])
                selected = "atomic" if identifier == "granularity" else "equivalent"
                answers[identifier] = {"type": "choice", "choice": selected,
                    "probabilities": {key: 0.99 if key == selected else 0.01 / (len(options) - 1) for key in options},
                    "confidence": 0.99}
        return TransportResponse(200, {}, json.dumps({"model": "jev-1.13.0", "answers": answers,
            "usage": {"input_tokens": 8, "output_tokens": 4}}).encode())


class InformationExecutionTests(unittest.TestCase):
    def run_native(self, tracker, *, concurrency=2, max_calls=100, attempts=1, progress=None):
        generator = OpenAIAdapter(config=LLMConfig(provider="openai", model="gpt-6-luna",
            api_key="offline-private-key", max_attempts=attempts, retry_backoff_seconds=0), sleep=Mock())
        evaluator = TypeSafeEvaluator(TypeSafeConfig(api_key="offline-private-evaluator"))
        analyzer = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
            concurrency=concurrency, qa_enabled=False, max_coverage_rounds=0, max_calls=max_calls),
            progress=progress, progress_interval_seconds=0.01)
        self.addCleanup(generator.close)
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=tracker.client)}), patch(
            "open_video_summary.adapters.typesafe._stdlib_transport", tracker.evaluate
        ):
            # The parent evaluator is constructed before the patch; replace its
            # stateless transport without disabling the native fork capability.
            evaluator.transport = tracker.evaluate
            report = analyzer.analyze(capture_snapshot(videos(["Backup diário."] * 3)))
        return report, generator, evaluator

    def test_native_workers_overlap_are_bounded_and_have_private_clients_and_records(self):
        tracker = _Requests(synchronize=True, delays={"v0:s0": 0.02})
        report, generator, evaluator = self.run_native(tracker)
        self.assertEqual("completed", report.status)
        self.assertEqual(2, tracker.maximum)
        self.assertEqual(3, report.counts.occurrences)
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(["c0", "t1:c0", "t2:c0"], [item.id for item in report.candidates])
        self.assertEqual({("v0:s0", "c0"), ("v0:s1", "t1:c0"), ("v0:s2", "t2:c0")}, set(tracker.evaluation_ids))
        self.assertEqual([], generator.records)
        self.assertEqual(3, len(evaluator.records))  # Only sequential pair decisions.
        self.assertEqual(4, len(tracker.clients))  # Parent preflight plus three workers.
        self.assertFalse(tracker.clients[0].closed)
        self.assertTrue(all(client.closed for client in tracker.clients[1:]))
        self.assertTrue(all(client.settings["max_retries"] == 0 for client in tracker.clients))
        self.assertTrue(all(client.settings["timeout"] == generator.config.timeout_seconds for client in tracker.clients))
        self.assertEqual(12, len(report.calls))
        for call in report.calls:
            metadata = json.loads(call.metadata_json)
            self.assertEqual(1, len(metadata["attempt_records"]))

    def test_order_and_inventory_are_stable_across_scheduling_and_serial_execution(self):
        one, _, _ = self.run_native(_Requests(delays={"v0:s0": 0.03}), concurrency=2)
        two, _, _ = self.run_native(_Requests(delays={"v0:s2": 0.03}), concurrency=2)
        serial, _, _ = self.run_native(_Requests(), concurrency=1)
        for field in ("candidates", "units", "occurrences", "relations", "coverage", "counts", "issues"):
            self.assertEqual(getattr(one, field), getattr(two, field), field)
            self.assertEqual(getattr(one, field), getattr(serial, field), field)
        self.assertEqual([item.input_hash for item in one.calls], [item.input_hash for item in serial.calls])

    def test_shared_logical_budget_bounds_parallel_work_and_retry_attempts_are_separate(self):
        tracker = _Requests(synchronize=True, transient=True)
        report, _, _ = self.run_native(tracker, max_calls=2, attempts=2)
        self.assertEqual(2, json.loads(report.metadata_json)["logical_calls"])
        self.assertEqual(2, len(report.calls))
        self.assertEqual(4, len(tracker.targets))
        self.assertEqual(2, tracker.maximum)
        self.assertEqual("failed", report.status)
        self.assertFalse(report.counts.valid_zero)
        self.assertIn("call_budget_exhausted", [item.kind for item in report.issues])
        self.assertTrue(all(len(json.loads(call.metadata_json)["attempt_records"]) == 2 for call in report.calls))
        self.assertTrue(all(client.closed for client in tracker.clients[1:]))

    def test_permanent_provider_failure_stops_dispatch_but_target_input_failure_does_not(self):
        tracker = _Requests(failures={"v0:s0": ExternalStatusError(401)})
        report, _, _ = self.run_native(tracker, concurrency=1)
        self.assertEqual(["v0:s0"], tracker.targets)
        self.assertEqual("AuthenticationError", json.loads(report.metadata_json)["permanent_provider_failure"])
        self.assertEqual("failed", report.status)
        target_error = ExternalStatusError(400, body={"error": {"param": "input"}})
        report, _, _ = self.run_native(_Requests(failures={"v0:s0": target_error}), concurrency=1)
        self.assertIsNone(json.loads(report.metadata_json)["permanent_provider_failure"])
        self.assertEqual(2, report.counts.occurrences)
        self.assertEqual("partial", report.status)

    def test_typesafe_endpoint_failure_stops_global_dispatch_but_target_400_does_not(self):
        tracker = _Requests(evaluation_failures={f"v0:s{index}": 404 for index in range(3)})
        report, _, _ = self.run_native(tracker, concurrency=1)
        self.assertEqual(["v0:s0"], tracker.targets)
        self.assertEqual([("v0:s0", "c0")], tracker.evaluation_ids)
        self.assertEqual(2, len(report.calls))
        self.assertEqual("ProviderConfigurationError", json.loads(report.metadata_json)["permanent_provider_failure"])
        self.assertEqual("failed", report.status)
        self.assertFalse(report.counts.valid_zero)
        tracker = _Requests(evaluation_failures={"v0:s0": 400})
        report, _, _ = self.run_native(tracker, concurrency=1)
        self.assertEqual(["v0:s0", "v0:s1", "v0:s2"], tracker.targets)
        self.assertIsNone(json.loads(report.metadata_json)["permanent_provider_failure"])
        self.assertEqual(2, report.counts.occurrences)
        self.assertEqual("partial", report.status)

    def test_concurrent_interrupt_stops_active_target_chains_and_queued_targets(self):
        for interrupted in ("v0:s0", "v0:s1"):
            with self.subTest(interrupted=interrupted):
                pending = "v0:s1" if interrupted == "v0:s0" else "v0:s0"
                tracker = _Requests(synchronize=True, delays={pending: 0.05},
                    failures={interrupted: KeyboardInterrupt()})
                with self.assertRaises(KeyboardInterrupt):
                    self.run_native(tracker)
                self.assertEqual({"v0:s0", "v0:s1"}, set(tracker.targets))
                self.assertEqual(2, len(tracker.targets))
                self.assertEqual([], tracker.evaluation_ids)
                self.assertTrue(all(client.closed for client in tracker.clients[1:]))
                self.assertFalse(any(thread.name.startswith("ovs-information")
                                     for thread in threading.enumerate()))

    def test_concurrent_interrupt_prevents_a_pending_native_request_from_retrying(self):
        tracker = _Requests(synchronize=True, delays={"v0:s1": 0.05}, transient=True,
            failures={"v0:s0": KeyboardInterrupt()})
        with self.assertRaises(KeyboardInterrupt):
            self.run_native(tracker, attempts=3)
        self.assertEqual({"v0:s0": 1, "v0:s1": 1}, tracker.target_attempts)
        self.assertEqual([], tracker.evaluation_ids)
        self.assertTrue(all(client.closed for client in tracker.clients[1:]))

    def test_injected_client_falls_back_to_serial_without_sharing_request_state(self):
        tracker = _Requests()
        generator = OpenAIAdapter(config=LLMConfig(provider="openai"), client=tracker.client())
        evaluator = TypeSafeEvaluator(TypeSafeConfig(api_key="fixture"), transport=tracker.evaluate)
        report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(qa_enabled=False)).analyze(
            capture_snapshot(videos(["Backup diário."] * 2)))
        self.assertEqual("completed", report.status)
        self.assertEqual(1, tracker.maximum)
        self.assertEqual(1, json.loads(report.metadata_json)["actual_concurrency"])

    def test_progress_waits_are_sanitized_and_stop_after_success_failure_and_interrupt(self):
        for failure in (None, ServiceTimeoutError("private transcript"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                events = []

                def slow(data, route):
                    time.sleep(0.045)
                    if failure is not None:
                        raise failure
                    return []

                analyzer = InformationAnalyzer(ScriptedGenerator(callback=slow), SyntheticEvaluator(),
                    InformationAnalysisConfig(qa_enabled=False), progress=events.append,
                    progress_interval_seconds=0.01)
                if isinstance(failure, KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        analyzer.analyze(capture_snapshot(videos(["private transcript"])))
                else:
                    analyzer.analyze(capture_snapshot(videos(["private transcript"])))
                self.assertTrue(any(event.event == "waiting" for event in events))
                before = len(events)
                time.sleep(0.025)
                self.assertEqual(before, len(events))
                self.assertFalse(any(thread.name == "ovs-information-progress" for thread in threading.enumerate()))
                self.assertNotIn("private transcript", repr(events))

    def test_observer_errors_are_isolated_and_library_default_does_not_print_progress(self):
        source = videos(["Backup diário."])
        items = {("v0:s0", "direct"): [candidate("v0:s0", source[0].segments[0].content)]}
        quiet = InformationAnalyzer(ScriptedGenerator(items), SyntheticEvaluator(), InformationAnalysisConfig(qa_enabled=False))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            baseline = quiet.analyze(capture_snapshot(source))
        self.assertEqual("", output.getvalue())
        noisy = InformationAnalyzer(ScriptedGenerator(items), SyntheticEvaluator(), InformationAnalysisConfig(qa_enabled=False),
            progress=Mock(side_effect=RuntimeError("broken observer")))
        report = noisy.analyze(capture_snapshot(source))
        self.assertEqual(baseline.units, report.units)
        self.assertEqual(baseline.counts, report.counts)
        self.assertEqual("completed", report.status)

    def test_report_observer_runs_before_selection_and_cannot_change_selection_on_failure(self):
        observed = []

        class Criterion:
            name = "Introduction"

            def evaluate(self, handler):
                observed.append("selection")
                return handler

        def observer(report, path):
            self.assertEqual("completed", report.status)
            observed.append("report")
            raise RuntimeError("observer failed")

        analyzer = InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator(), InformationAnalysisConfig(qa_enabled=False))
        Summarizer([Criterion()]).summarize(videos(["Bom dia."]), save_output=False,
            information_analyzer=analyzer, information_report_observer=observer)
        self.assertEqual(["report", "selection"], observed)


class RetryExecutionTests(unittest.TestCase):
    def test_typesafe_404_is_provider_wide_and_400_remains_a_target_request_failure(self):
        for status, error_class in ((404, ProviderConfigurationError), (400, ConfigurationError)):
            with self.subTest(status=status):
                transport = Mock(return_value=TransportResponse(status, {}, b"private"))
                evaluator = TypeSafeEvaluator(TypeSafeConfig(api_key="private-key"),
                    transport=transport, sleep=Mock())
                with self.assertRaises(error_class) as caught:
                    evaluator.evaluate("private transcript", noul={"q": "Question?"})
                self.assertIs(error_class, type(caught.exception))
                self.assertEqual(1, transport.call_count)
                evaluator.sleep.assert_not_called()
                self.assertNotIn("private", str(caught.exception))

    def test_openai_retry_after_and_capped_jitter_with_attempt_progress(self):
        error = ExternalStatusError(429)
        error.response = SimpleNamespace(headers={"Retry-After": "3"})
        create = Mock(side_effect=[error, TimeoutError("private"), response("ok")])
        waits, events = [], []
        adapter = OpenAIAdapter(config=LLMConfig(provider="openai", retry_backoff_seconds=1),
            client=SimpleNamespace(responses=SimpleNamespace(create=create)), sleep=waits.append,
            progress=events.append, jitter=lambda low, high: high)
        result = adapter.generate(GenerationRequest("private transcript"))
        self.assertEqual([3.0, 2.5], waits)
        self.assertEqual(3, result.metadata.attempts)
        self.assertEqual(3, create.call_count)
        self.assertEqual(3, len([event for event in events if event.event == "attempt_started"]))
        self.assertEqual(["RateLimitError", "ServiceTimeoutError"], [event.error_type for event in events if event.event == "attempt_failed"])
        self.assertNotIn("private", repr(events))
        self.assertEqual(30.0, retry_after({"retry-after": "900"}))
        self.assertEqual(10.0, retry_delay(8, 1, backoff_cap=10, jitter=lambda low, high: high))

    def test_typesafe_transient_retry_honors_server_delay_and_permanent_auth_is_not_retried(self):
        waits, events = [], []
        good = TransportResponse(200, {}, b'{"model":"jev-1.13.0","answers":{"q":{"type":"noul","noul":0.7}}}')
        transport = Mock(side_effect=[TransportResponse(529, {"Retry-After": "2"}, b"private"), good])
        evaluator = TypeSafeEvaluator(TypeSafeConfig(api_key="private-key", retry_backoff_seconds=0.5),
            transport=transport, sleep=waits.append, progress=events.append, jitter=lambda low, high: high)
        self.assertEqual(0.7, evaluator.evaluate("private transcript", noul={"q": "Question?"}).noul[0].probability)
        self.assertEqual([2.0], waits)
        self.assertEqual([2.0], [event.delay_seconds for event in events if event.event == "retry_scheduled"])
        self.assertNotIn("private", repr(events))
        denied = Mock(return_value=TransportResponse(401, {}, b"private"))
        evaluator = TypeSafeEvaluator(TypeSafeConfig(api_key="private-key"), transport=denied, sleep=Mock())
        with self.assertRaises(AuthenticationError):
            evaluator.evaluate("private transcript", noul={"q": "Question?"})
        self.assertEqual(1, denied.call_count)
        evaluator.sleep.assert_not_called()


class ExecutionConfigurationAndCLITests(unittest.TestCase):
    def test_concurrency_defaults_bounds_and_explicit_process_file_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / ".env"
            fixture.write_text("OVS_INFORMATION_CONCURRENCY=3\n", encoding="utf-8")
            self.assertEqual(3, configured_information_analyzer(environ={}, env_file=fixture).config.concurrency)
            self.assertEqual(4, configured_information_analyzer(environ={"OVS_INFORMATION_CONCURRENCY": "4"}, env_file=fixture).config.concurrency)
            observer = Mock()
            analyzer = configured_information_analyzer({"information_concurrency": 1},
                environ={"OVS_INFORMATION_CONCURRENCY": "4"}, env_file=fixture, progress=observer)
            self.assertEqual(1, analyzer.config.concurrency)
            self.assertIs(observer, analyzer.progress)
            self.assertEqual(2, configured_information_analyzer(environ={}, env_file=Path(temporary) / "missing.env").config.concurrency)
            for value in (0, 9, -1, True, 1.5):
                with self.subTest(value=value), self.assertRaises(ConfigurationError):
                    InformationAnalysisConfig(concurrency=value)

    def test_both_cli_commands_select_concurrency_and_progress_messages_are_flushed(self):
        for command in ("summarize", "analyze-information"):
            args = cli.build_parser().parse_args([command, "--information-concurrency", "4"])
            self.assertEqual(4, args.information_concurrency)
        events = [
            AnalysisProgress("fixture", "analysis_started", 0, 0, 20, target_total=2, concurrency=2),
            AnalysisProgress("fixture", "provider_attempt", 1, 1, 20, target_index=0, target_total=2,
                segment_id="v0:s0", operation="extract_direct", provider="openai",
                service=ProviderProgress("retry_scheduled", "openai", 1, 3, delay_seconds=2)),
            AnalysisProgress("fixture", "waiting", 11, 1, 20, operation="extract_direct",
                provider="openai", operation_seconds=11),
        ]
        with patch("builtins.print") as printer:
            for event in events:
                cli._print_information_progress(event)
            cli._print_information_report(None, None)
            report = InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator(), InformationAnalysisConfig(qa_enabled=False)).analyze(capture_snapshot([]))
            cli._print_information_report(report, "outputs/information/fixture.json")
        self.assertTrue(all(call.kwargs.get("flush") for call in printer.call_args_list))
        text = " ".join(call.args[0] for call in printer.call_args_list)
        self.assertIn("concurrency 2", text)
        self.assertIn("retry scheduled", text)
        self.assertIn("still running", text)
        self.assertIn("Information report", text)


if __name__ == "__main__":
    unittest.main()
