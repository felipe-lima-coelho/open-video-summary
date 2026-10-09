"""Cold, offline response replay for bounded independent discovery lookahead."""

import json
import threading
import time
import unittest
from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

from open_video_summary.adapters.llm import DomainResponseInterpreter
from open_video_summary.adapters.typesafe import (
    ChoiceResult, EvaluationMetadata, EvaluationResult, NoulResult,
)
from open_video_summary.contracts import GenerationResult, ServiceMetadata
from open_video_summary.core.summarizers.information_analysis import (
    InformationAnalyzer, _AnalysisRun,
)
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import capture_snapshot
from open_video_summary.errors import AuthenticationError, ServiceTimeoutError
from open_video_summary.utils.providers import LLMConfig
from test_information_analysis import candidate, videos


SENTENCES = (
    "A primeira regra estabelece backup automático diário com uma cópia no servidor remoto.",
    "A segunda regra exige retenção de trinta dias para cada cópia do servidor remoto.",
    "A terceira regra permite arquivar as cópias antigas em um sistema separado do servidor.",
)
SOURCE = " ".join(SENTENCES)


class _Tape:
    def __init__(self, *, failure=None, barrier=False, delay=.015, other_delay=None):
        self.lock = threading.Lock()
        self.active = self.maximum = self.generator_active = self.generator_maximum = 0
        self.requests, self.evaluations, self.instances = [], [], []
        self.failure, self.barrier, self.delay = failure, barrier, delay
        self.other_delay = other_delay
        self.qa_started, self.focus_started = threading.Event(), threading.Event()

    def enter(self, generator=False):
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            if generator:
                self.generator_active += 1
                self.generator_maximum = max(self.generator_maximum, self.generator_active)

    def leave(self, generator=False):
        with self.lock:
            self.active -= 1
            self.generator_active -= int(generator)


class _Generator:
    can_fork = True

    def __init__(self, tape):
        self.tape, self.records, self.closed = tape, [], False
        self.config = LLMConfig(provider="openai", model="fixture")
        self.progress = self.cancel_event = None
        tape.instances.append(self)

    def preflight(self):
        pass

    def fork(self):
        return type(self)(self.tape)

    def close(self):
        self.closed = True

    def generate(self, request):
        data = json.loads(request.prompt.split("\nInput:\n", 1)[1])
        route = data.get("analysis_task") or (
            "qa" if request.output.kind == "information_qa" else
            "recovery" if "coverage_audit" in data else "direct")
        target = data["target"]
        window = data.get("discovery_window")
        first = window is not None and window["start_char"] == 0 and window["end_char"] > 80
        with self.tape.lock:
            self.tape.requests.append((request.prompt, asdict(request.output), request.temperature))
        self.tape.enter(True)
        started = time.monotonic()
        error = None
        try:
            if route == "qa" and first:
                self.tape.qa_started.set()
                if self.tape.barrier and target["id"] == "v0:s0" and not self.tape.focus_started.wait(2):
                    raise AssertionError("Independent first focus did not overlap QA")
            if route == "discover_coverage_foci":
                self.tape.focus_started.set()
                if self.tape.barrier and not self.tape.qa_started.wait(2):
                    raise AssertionError("Independent first QA did not overlap focus")
            time.sleep(self.tape.delay if target["id"] == "v0:s0" or self.tape.other_delay is None
                       else self.tape.other_delay)
            if target["id"] == "v0:s0":
                if self.tape.failure == "qa_timeout" and route == "qa" and first:
                    raise ServiceTimeoutError("offline")
                if self.tape.failure == "focus_timeout" and route == "discover_coverage_foci":
                    raise ServiceTimeoutError("offline")
                if self.tape.failure == "authentication" and route == "qa" and first:
                    raise AuthenticationError("offline")
                if self.tape.failure == "interrupt" and route == "qa" and first:
                    raise KeyboardInterrupt()
            source = target["text"]
            rows = [candidate(target["id"], source, quote=sentence, qa=route == "qa")
                    for sentence in SENTENCES if sentence in source and
                    (window is None or window["start_char"] <= source.index(sentence) < window["end_char"])]
            envelope = {"candidates": rows, "issues": []}
            text = json.dumps(envelope, ensure_ascii=False)
            value = DomainResponseInterpreter().interpret(text, request.output)
        except BaseException as exc:
            error = exc
            raise
        finally:
            metadata = ServiceMetadata("openai", "fixture", returned_model="fixture",
                duration_seconds=time.monotonic() - started,
                status=type(error).__name__ if error is not None else "completed",
                input_tokens=17, output_tokens=11)
            self.records.append(metadata)
            self.tape.leave(True)
        return GenerationResult(text, value, metadata)


class _Evaluator:
    can_fork = True

    def __init__(self, tape):
        self.tape, self.records, self.closed = tape, [], False
        self.config = SimpleNamespace(provider="fixture", model="fixture")
        self.progress = self.cancel_event = None
        tape.instances.append(self)

    def preflight(self):
        pass

    def fork(self):
        return type(self)(self.tape)

    def close(self):
        self.closed = True

    def evaluate(self, context, noul=None, choice=None):
        noul, choice = noul or {}, choice or {}
        with self.tape.lock:
            self.tape.evaluations.append((context, dict(noul), dict(choice)))
        self.tape.enter()
        try:
            time.sleep(.001)
            probabilities = {key: .01 if context.startswith("Original target [") else .99
                             for key in noul}
            choices = []
            for key, (_, options) in choice.items():
                selected = "atomic" if key == "granularity" else "complementary"
                choices.append(ChoiceResult(key, selected,
                    tuple((option, .99 if option == selected else .01 / (len(options) - 1))
                          for option in options), .99))
            metadata = EvaluationMetadata("fixture", "fixture", .001, 1, "success", 20, 8,
                                          provider="fixture")
            self.records.append(metadata)
            return EvaluationResult(tuple(NoulResult(key, value) for key, value in probabilities.items()),
                                    tuple(choices), metadata)
        finally:
            self.tape.leave()


class InformationDiscoveryParallelTests(unittest.TestCase):
    def replay(self, *, lookahead=True, failure=None, barrier=False, max_calls=1000,
               settings=None, delay=.015, texts=None, other_delay=None):
        tape = _Tape(failure=failure, barrier=barrier, delay=delay, other_delay=other_delay)
        generator, evaluator = _Generator(tape), _Evaluator(tape)
        config = InformationAnalysisConfig(concurrency=2, direct_window_chars=160,
            qa_window_chars=160, max_pair_comparisons=0, **(settings or {}), max_calls=max_calls)
        analyzer = InformationAnalyzer(generator, evaluator, config)
        started = time.monotonic()
        if lookahead:
            report = analyzer.analyze(capture_snapshot(videos(texts or [SOURCE, SOURCE])))
        else:
            with patch.object(_AnalysisRun, "_seed_discovery", return_value=None):
                report = analyzer.analyze(capture_snapshot(videos(texts or [SOURCE, SOURCE])))
        return report, tape, time.monotonic() - started

    def assert_same_semantics(self, serial, parallel):
        for field in ("status", "candidates", "units", "occurrences", "relations", "coverage",
                      "counts", "issues", "decompositions", "coverage_foci"):
            self.assertEqual(getattr(serial, field), getattr(parallel, field), field)
        self.assertEqual([(call.operation, call.input_hash, call.output_hash, call.status, call.decisions_json)
                          for call in serial.calls],
                         [(call.operation, call.input_hash, call.output_hash, call.status, call.decisions_json)
                          for call in parallel.calls])
        a, b = json.loads(serial.metadata_json), json.loads(parallel.metadata_json)
        for field in ("qa_source_windows", "direct_source_windows", "discovery_window_declarations",
                      "logical_calls", "physical_attempts", "retry_attempts", "prompt_template_hashes"):
            self.assertEqual(a[field], b[field], field)

    def test_cold_replay_preserves_exact_prompts_records_and_order(self):
        serial, old, _ = self.replay(lookahead=False)
        parallel, new, _ = self.replay()
        self.assert_same_semantics(serial, parallel)
        self.assertEqual(Counter((prompt, json.dumps(spec, sort_keys=True), temperature)
                                 for prompt, spec, temperature in old.requests),
                         Counter((prompt, json.dumps(spec, sort_keys=True), temperature)
                                 for prompt, spec, temperature in new.requests))
        self.assertEqual(Counter(json.dumps(item, sort_keys=True) for item in old.evaluations),
                         Counter(json.dumps(item, sort_keys=True) for item in new.evaluations))
        stats = json.loads(parallel.metadata_json)["discovery_lookahead"]
        self.assertEqual((4, 4, 0), (stats["started"], stats["consumed"], stats["discarded"]))
        self.assertLessEqual(new.maximum, 2)
        self.assertEqual(0, new.active)
        self.assertTrue(all(item.closed for item in new.instances[2:]))
        self.assertEqual([], new.instances[0].records)
        self.assertEqual([], new.instances[1].records)

    def test_independent_first_requests_overlap_and_reduce_controlled_wall_time(self):
        serial, _, serial_seconds = self.replay(lookahead=False, delay=.07, other_delay=0,
                                               texts=[SOURCE, "Bom dia."])
        parallel, tape, parallel_seconds = self.replay(barrier=True, delay=.07, other_delay=0,
                                                      texts=[SOURCE, "Bom dia."])
        self.assert_same_semantics(serial, parallel)
        self.assertEqual(2, tape.generator_maximum)
        self.assertLess(parallel_seconds, serial_seconds - .035)
        self.assertLessEqual(tape.maximum, 2)

    def test_timeout_keeps_depth_first_window_recovery_and_failed_focus_audit(self):
        for failure in ("qa_timeout", "focus_timeout"):
            with self.subTest(failure=failure):
                serial, old, _ = self.replay(lookahead=False, failure=failure)
                parallel, new, _ = self.replay(failure=failure)
                self.assert_same_semantics(serial, parallel)
                self.assertEqual(Counter(prompt for prompt, _, _ in old.requests),
                                 Counter(prompt for prompt, _, _ in new.requests))
                self.assertIn("ServiceTimeoutError", [call.status for call in parallel.calls])
                self.assertTrue(parallel.coverage)

    def test_tight_budget_retains_original_admission_without_extra_requests(self):
        for maximum in (1, 3, 12, 40):
            with self.subTest(max_calls=maximum):
                serial, old, _ = self.replay(lookahead=False, max_calls=maximum)
                parallel, new, _ = self.replay(max_calls=maximum)
                # Existing target concurrency can choose different surviving
                # target records at exhaustion; request/attempt bounds remain.
                stats = json.loads(parallel.metadata_json)["discovery_lookahead"]
                self.assertEqual(0, stats["submitted"])
                self.assertEqual(len(old.requests), len(new.requests))
                self.assertLessEqual(json.loads(parallel.metadata_json)["logical_calls"], maximum)
                self.assertEqual(json.loads(serial.metadata_json)["physical_attempts"],
                                 json.loads(parallel.metadata_json)["physical_attempts"])

    def test_prefix_boundary_counts_adaptive_attempts_and_both_seed_reservations(self):
        for remaining, expected in ((143, 0), (144, 1), (580, 1), (581, 2)):
            with self.subTest(remaining=remaining):
                tape = _Tape(delay=0)
                config = InformationAnalysisConfig(concurrency=2, max_calls=remaining)
                root = _AnalysisRun(InformationAnalyzer(_Generator(tape), _Evaluator(tape), config),
                                    capture_snapshot(videos([SOURCE])))
                root.active_target = "v0:s0"
                root.source_attempts["direct"] = 1
                root.discovery = SimpleNamespace(disable=lambda reason: None,
                    submit=lambda prompt, spec, reservation, callback:
                        SimpleNamespace(prompt=prompt, spec=spec, reservation=reservation))
                root._seed_discovery(root.snapshot.current_segments[0], "direct", 0, (0, 160), 4)
                self.assertEqual(expected, len(root.lookahead))
                self.assertEqual(expected, root.budget.reserved)
                for item in root.lookahead.values():
                    item.reservation.release()
                self.assertEqual(0, root.budget.reserved)

    def test_focus_only_guard_retains_the_prepaid_discovery_admission_margin(self):
        for remaining, expected in ((4, 0), (5, 1)):
            with self.subTest(remaining=remaining):
                tape = _Tape(delay=0)
                config = InformationAnalysisConfig(concurrency=2, max_calls=remaining,
                    qa_enabled=False, max_literal_repairs=0, max_granularity_checks=0)
                root = _AnalysisRun(InformationAnalyzer(_Generator(tape), _Evaluator(tape), config),
                                    capture_snapshot(videos(["Bom dia."])))
                root.active_target = "v0:s0"
                root.discovery = SimpleNamespace(disable=lambda reason: None,
                    submit=lambda prompt, spec, reservation, callback:
                        SimpleNamespace(prompt=prompt, spec=spec, reservation=reservation))
                root._seed_discovery(root.snapshot.current_segments[0], "direct", 0, None, 0)
                self.assertEqual(expected, len(root.lookahead))
                self.assertEqual(expected, root.budget.reserved)
                self.assertGreater(root.budget.remaining, 3)
                for item in root.lookahead.values():
                    item.reservation.release()
                self.assertEqual(0, root.budget.reserved)

    def test_permanent_failure_is_visible_bounded_and_closes_started_workers(self):
        report, tape, _ = self.replay(failure="authentication", barrier=True)
        self.assertNotEqual("completed", report.status)
        self.assertIn("AuthenticationError", [call.status for call in report.calls])
        self.assertEqual("AuthenticationError", json.loads(report.metadata_json)["permanent_provider_failure"])
        self.assertLessEqual(tape.maximum, 2)
        self.assertEqual(0, tape.active)
        self.assertTrue(all(item.closed for item in tape.instances[2:]))
        self.assertFalse(any(thread.name.startswith("ovs-information-discovery")
                             for thread in threading.enumerate()))

    def test_interrupt_drains_all_futures_and_owned_adapters(self):
        tape = _Tape(failure="interrupt", barrier=True)
        config = InformationAnalysisConfig(concurrency=2, max_calls=1000,
            direct_window_chars=160, qa_window_chars=160, max_pair_comparisons=0)
        run = _AnalysisRun(InformationAnalyzer(_Generator(tape), _Evaluator(tape), config),
                           capture_snapshot(videos([SOURCE, SOURCE])))
        with self.assertRaises(KeyboardInterrupt):
            run.run()
        self.assertEqual(0, run.budget.reserved)
        self.assertEqual(0, tape.active)
        self.assertTrue(all(item.closed for item in tape.instances[2:]))
        self.assertFalse(any(thread.name.startswith("ovs-information-discovery")
                             for thread in threading.enumerate()))

    def test_disabled_routes_and_unforkable_adapters_send_no_seed_requests(self):
        report, tape, _ = self.replay(settings={"qa_enabled": False, "max_coverage_foci": 0})
        self.assertEqual({"enabled": False}, json.loads(report.metadata_json)["discovery_lookahead"])
        self.assertTrue(all('"analysis_task"' not in prompt and '"information_qa"' not in prompt
                            for prompt, _, _ in tape.requests))
        tape = _Tape(delay=0)
        generator, evaluator = _Generator(tape), _Evaluator(tape)
        generator.can_fork = False
        report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
            max_calls=1000, max_pair_comparisons=0)).analyze(
            capture_snapshot(videos([SOURCE, SOURCE])))
        self.assertEqual(1, json.loads(report.metadata_json)["actual_concurrency"])
        self.assertEqual({"enabled": False}, json.loads(report.metadata_json)["discovery_lookahead"])
        self.assertLessEqual(tape.maximum, 1)

    def test_forkable_stateless_generator_can_omit_optional_close(self):
        class StatelessGenerator:
            can_fork = True

            def __init__(self, tape):
                self.delegate = _Generator(tape)
                self.config, self.records = self.delegate.config, self.delegate.records

            def preflight(self):
                pass

            def fork(self):
                return type(self)(self.delegate.tape)

            def generate(self, request):
                return self.delegate.generate(request)

        reports = []
        for lookahead in (False, True):
            tape = _Tape(delay=0)
            generator, evaluator = StatelessGenerator(tape), _Evaluator(tape)
            self.assertFalse(hasattr(generator, "close"))
            config = InformationAnalysisConfig(concurrency=2, max_calls=1000,
                direct_window_chars=160, qa_window_chars=160, max_pair_comparisons=0)
            analyzer = InformationAnalyzer(generator, evaluator, config)
            snapshot = capture_snapshot(videos([SENTENCES[0], "Bom dia."]))
            if lookahead:
                report = analyzer.analyze(snapshot)
            else:
                with patch.object(_AnalysisRun, "_seed_discovery", return_value=None):
                    report = analyzer.analyze(snapshot)
            self.assertEqual("completed", report.status)
            self.assertNotIn("adapter_cleanup_failed", [issue.kind for issue in report.issues])
            reports.append(report)
        self.assert_same_semantics(*reports)
        self.assertEqual(4, json.loads(reports[1].metadata_json)["discovery_lookahead"]["consumed"])

    def test_callable_cleanup_failures_remain_visible_after_lookahead(self):
        closed = []

        def fail_close(generator):
            closed.append(generator)
            generator.closed = True
            raise RuntimeError("offline cleanup")

        tape = _Tape(delay=0)
        config = InformationAnalysisConfig(concurrency=2, max_calls=1000,
            direct_window_chars=160, qa_window_chars=160, max_pair_comparisons=0)
        with patch.object(_Generator, "close", fail_close):
            report = InformationAnalyzer(_Generator(tape), _Evaluator(tape), config).analyze(
                capture_snapshot(videos([SENTENCES[0], "Bom dia."])))
        self.assertEqual("partial", report.status)
        self.assertEqual(6, len(closed))  # Four lookaheads and two target-owned generators.
        self.assertTrue(any(issue.kind == "adapter_cleanup_failed" and issue.detail == "RuntimeError"
                            for issue in report.issues))
        self.assertEqual(4, json.loads(report.metadata_json)["discovery_lookahead"]["consumed"])


if __name__ == "__main__":
    unittest.main()
