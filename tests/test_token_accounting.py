"""Offline regression for measured Jev under-reservation and usage feedback."""

import json
import math
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from open_video_summary.adapters.typesafe import TypeSafeConfig, TypeSafeEvaluator
from open_video_summary.errors import ConfigurationError, InvalidResponseError, RequestDeadlineError
from open_video_summary.utils.request_control import RequestController, RequestLimits, estimate_input_tokens
from tests.test_request_control import FakeTime
from tests.test_typesafe_adapter import FakeTransport, response


class UsageAccountingTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeTime()
        self.control = RequestController(RequestLimits(tokens=80000, window_seconds=1), clock=self.clock.now)

    def admit(self, tokens=230, **kwargs):
        return self.control.admit(tokens, deadline=100, sleep=self.clock.sleep, **kwargs)

    def test_observed_230_to_294_charges_input_excess_and_adapts_next_request(self):
        first = self.admit(adaptive_estimate=True)
        feedback = self.control.complete(first, input_tokens=294, output_tokens=21)
        self.assertEqual((294, 64, True, "reported"),
            (feedback.accounted_tokens, feedback.token_adjustment, feedback.estimation_adjusted, feedback.usage_status))
        self.assertEqual(294, sum(item[2] for item in self.control._reservations))
        self.assertAlmostEqual(1.25 * 294 / 230, feedback.estimation_multiplier_after)
        second = self.admit(adaptive_estimate=True)
        self.assertEqual(368, second.reserved_tokens)
        below = self.control.complete(second, input_tokens=294, output_tokens=10000)
        self.assertEqual((368, 0, False), (below.accounted_tokens, below.token_adjustment, below.estimation_adjusted))
        self.assertEqual(662, sum(item[2] for item in self.control._reservations))
        self.control.complete(first, input_tokens=294, output_tokens=21)
        self.assertEqual(662, sum(item[2] for item in self.control._reservations))

    def test_actual_excess_consumes_capacity_and_queued_prediction_is_recomputed(self):
        self.control = RequestController(RequestLimits(tokens=600, window_seconds=1), clock=self.clock.now)
        first = self.admit(adaptive_estimate=True)
        def while_waiting(seconds):
            self.control.complete(first, input_tokens=294)
            self.clock.sleep(seconds)
        # Feedback makes this previously fitting request exceed the single-window cap.
        with self.assertRaises(ConfigurationError):
            self.control.admit(400, adaptive_estimate=True, deadline=100, sleep=while_waiting)
        self.assertEqual(1, self.control._sequence)
        self.assertEqual([1], self.clock.waits)

    def test_expired_admission_excess_becomes_token_debt_without_another_request(self):
        first = self.admit()
        self.clock.value = 2
        feedback = self.control.complete(first, input_tokens=294, output_tokens=21)
        self.assertEqual(64, feedback.token_adjustment)
        self.assertEqual([2, first.sequence, 64, False], self.control._reservations[-1])
        self.control._tokens = 100
        self.control._requests = 1
        admission = self.admit(50)
        self.assertEqual(1, admission.wait_seconds)
        self.assertIn("token_window", admission.wait_reasons)
        self.assertNotIn("request_window", admission.wait_reasons)

    def test_parallel_feedback_never_loses_charges_or_reduces_margin(self):
        admissions = [self.admit(adaptive_estimate=True) for _ in range(8)]
        observed = [294, 310, 400, 600, 280, 450, 500, 350]
        barrier = threading.Barrier(8)
        def finish(pair):
            barrier.wait(timeout=2)
            return self.control.complete(pair[0], input_tokens=pair[1], output_tokens=21)
        with ThreadPoolExecutor(max_workers=8) as workers:
            feedback = list(workers.map(finish, zip(admissions, observed)))
        self.assertEqual(sum(observed), sum(item[2] for item in self.control._reservations))
        expected = 1.25 * max(observed) / 230
        self.assertAlmostEqual(expected, self.control.snapshot()["input_estimation_multiplier"])
        self.assertEqual(math.ceil(230 * expected), self.admit(adaptive_estimate=True).reserved_tokens)
        self.assertEqual(observed, [item.accounted_tokens for item in feedback])

    def test_invalid_or_missing_counts_do_not_corrupt_capacity(self):
        for value in (-1, 1.5, float("nan"), float("inf"), True, 2 ** 63):
            with self.subTest(value=value):
                first = self.admit(adaptive_estimate=True)
                feedback = self.control.complete(first, input_tokens=value)
                self.assertEqual((230, 0, "invalid"),
                    (feedback.accounted_tokens, feedback.token_adjustment, feedback.usage_status))
        unknown = self.control.complete(self.admit(), input_tokens=None)
        self.assertEqual("unknown", unknown.usage_status)
        self.assertEqual(1, self.control.snapshot()["input_estimation_multiplier"])
        self.assertEqual(7 * 230, sum(item[2] for item in self.control._reservations))


class JevPredictionTests(unittest.TestCase):
    @staticmethod
    def observed_context():
        # This serialized request is exactly 166 UTF-8 bytes: legacy reserve 230.
        payload = {"state": "", "model": "jev-1.13.0", "questions": {"q": {"type": "noul", "instructions": "Question?"}}}
        empty = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        return "A" * (166 - len(empty))

    def evaluator(self, transport, **settings):
        clock = FakeTime()
        evaluator = TypeSafeEvaluator(TypeSafeConfig(api_key="offline", max_attempts=1, **settings),
            transport=transport, sleep=clock.sleep, jitter=lambda *_: 0)
        evaluator.controller = RequestController(RequestLimits(tokens=80000, window_seconds=1), clock=clock.now)
        return evaluator, clock

    def test_defaults_cover_observed_usage_and_legacy_zero_allowance_adapts(self):
        for overrides, expected in (({}, 614), ({"token_request_overhead": 0, "token_question_overhead": 0}, 230)):
            with self.subTest(overrides=overrides):
                good = response({"q": {"type": "noul", "noul": .99}}, usage={"input_tokens": 294, "output_tokens": 21})
                transport = FakeTransport(good, good)
                evaluator, _ = self.evaluator(transport, **overrides)
                first = evaluator.evaluate(self.observed_context(), noul={"q": "Question?"}).metadata
                self.assertEqual(230, estimate_input_tokens(transport.calls[0]["body"]))
                self.assertEqual(expected, first.reserved_tokens)
                self.assertEqual(max(expected, 294), first.accounted_tokens)
                self.assertEqual(294, first.input_tokens)
                self.assertEqual(21, first.output_tokens)
                self.assertEqual("reported", first.usage_status)
                self.assertEqual(bool(overrides), first.estimation_adjusted)
                second = evaluator.evaluate(self.observed_context(), noul={"q": "Question?"}).metadata
                self.assertEqual(368 if overrides else 614, second.reserved_tokens)
                self.assertEqual(0, second.token_adjustment)

    def test_small_large_states_and_multiple_question_criteria_remain_in_prediction(self):
        for state in ("Synthetic state", "Português: questão. " * 500):
            for count in (1, 6):
                with self.subTest(state_size=len(state), count=count):
                    def transport(url, *, body, **kwargs):
                        payload = json.loads(body)
                        answers = {key: {"type": "choice", "choice": "equivalent",
                            "probabilities": {"equivalent": .99, "different": .01}}
                            for key in payload["questions"]}
                        predicted = estimate_input_tokens(body) + 256 + 128 * count
                        return response(answers, usage={"input_tokens": predicted - 10, "output_tokens": 21})
                    recorded = Mock(side_effect=transport)
                    evaluator, _ = self.evaluator(recorded)
                    criteria = {"equivalent": "Same precise fact", "different": "A different precise fact"}
                    result = evaluator.evaluate(state, choice={f"p{i}": ("Compare candidates?", criteria) for i in range(count)})
                    payload = recorded.call_args.kwargs["body"]
                    self.assertEqual(state, json.loads(payload)["state"])
                    self.assertEqual(criteria, json.loads(payload)["questions"]["p0"]["criteria"])
                    self.assertEqual(estimate_input_tokens(payload) + 256 + 128 * count, result.metadata.reserved_tokens)
                    self.assertEqual(count, result.metadata.question_count)
                    self.assertEqual(result.metadata.reserved_tokens, result.metadata.accounted_tokens)

    def test_invalid_semantics_headers_or_elapsed_deadline_still_charge_reported_input(self):
        for failure in ("semantics", "headers", "deadline"):
            with self.subTest(failure=failure):
                answers = {} if failure == "semantics" else {"q": {"type": "noul", "noul": .99}}
                good = response(answers, headers={"bad": 123} if failure == "headers" else {},
                    usage={"input_tokens": 294, "output_tokens": 21})
                evaluator, clock = self.evaluator(Mock(return_value=good), token_request_overhead=0, token_question_overhead=0)
                if failure == "deadline":
                    def delayed(*args, **kwargs):
                        clock.value = 100
                        return good
                    evaluator.transport = delayed
                with self.assertRaises(RequestDeadlineError if failure == "deadline" else InvalidResponseError):
                    evaluator.evaluate(self.observed_context(), noul={"q": "Question?"})
                metadata = evaluator.records[-1]
                self.assertEqual((294, 64, "reported", True),
                    (metadata.accounted_tokens, metadata.token_adjustment, metadata.usage_status, metadata.request_sent))
                self.assertEqual(294, metadata.input_tokens)

    def test_callback_failure_does_not_skip_charge(self):
        good = response({"q": {"type": "noul", "noul": .99}}, usage={"input_tokens": 294, "output_tokens": 21})
        evaluator, _ = self.evaluator(FakeTransport(good), token_request_overhead=0, token_question_overhead=0)
        evaluator.progress = Mock(side_effect=RuntimeError("offline callback"))
        result = evaluator.evaluate(self.observed_context(), noul={"q": "Question?"})
        self.assertEqual(294, result.metadata.accounted_tokens)
        self.assertEqual(64, result.metadata.token_adjustment)

    def test_invalid_input_usage_is_unknown_charge_and_invalid_output_retains_valid_input(self):
        for name in ("input_tokens", "output_tokens"):
            for value in (-1, .5, float("nan"), float("inf"), True, 2 ** 63):
                with self.subTest(name=name, value=value):
                    usage = {"input_tokens": 294, "output_tokens": 21, name: value}
                    evaluator, _ = self.evaluator(FakeTransport(response({"q": {"type": "noul", "noul": .99}}, usage=usage)),
                        token_request_overhead=0, token_question_overhead=0)
                    with self.assertRaises(InvalidResponseError):
                        evaluator.evaluate(self.observed_context(), noul={"q": "Question?"})
                    metadata = evaluator.records[-1]
                    self.assertEqual("invalid", metadata.usage_status)
                    self.assertEqual(294 if name == "output_tokens" else 230, metadata.accounted_tokens)
                    self.assertEqual(64 if name == "output_tokens" else 0, metadata.token_adjustment)
        evaluator, _ = self.evaluator(FakeTransport(response({"q": {"type": "noul", "noul": .99}})))
        result = evaluator.evaluate(self.observed_context(), noul={"q": "Question?"})
        self.assertEqual("unknown", result.metadata.usage_status)
        self.assertEqual(result.metadata.reserved_tokens, result.metadata.accounted_tokens)

    def test_overhead_configuration_is_bounded(self):
        for name in ("token_request_overhead", "token_question_overhead"):
            for value in (-1, 100001, 1.5, True):
                with self.subTest(name=name, value=value), self.assertRaises(ConfigurationError):
                    TypeSafeConfig(**{name: value})


if __name__ == "__main__":
    unittest.main()
