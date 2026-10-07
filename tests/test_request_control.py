"""Offline admission tests use a fake monotonic clock and actual adapter boundaries."""

import json
import threading
import time
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from open_video_summary.adapters.llm import OpenAIAdapter
from open_video_summary.adapters.typesafe import TypeSafeConfig, TypeSafeEvaluator, TransportResponse
from open_video_summary.contracts import GenerationRequest, OutputSpec
from open_video_summary.errors import (
    AuthenticationError, ConfigurationError, ProviderConfigurationError,
    RateLimitError, RequestCancelledError, RequestDeadlineError,
    ServiceTimeoutError, ServiceUnavailableError,
)
from open_video_summary.utils.providers import LLMConfig
from open_video_summary.utils.request_control import (
    RequestController, RequestLimits, RequestScope, estimate_input_tokens,
)
from open_video_summary.utils.retry import retry_after
from tests.test_llm_adapters import ExternalStatusError, response


class FakeTime:
    def __init__(self):
        self.value, self.waits = 0.0, []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.value += seconds


class RequestControlTests(unittest.TestCase):
    def controller(self, **limits):
        clock = FakeTime()
        return RequestController(RequestLimits(**limits), clock=clock.now), clock

    def admit(self, control, clock, tokens=100, **kwargs):
        return control.admit(tokens, deadline=kwargs.pop("deadline", 10000),
                             sleep=clock.sleep, **kwargs)

    def test_retry_after_seconds_date_ms_nonfinite_and_case(self):
        self.assertEqual(60, retry_after({"ReTrY-AfTeR": "60"}))
        self.assertEqual(900, retry_after({"Retry-After": "900"}, cap=30))
        now = datetime(2026, 10, 7, tzinfo=timezone.utc)
        self.assertEqual(60, retry_after({"Retry-After": "Wed, 07 Oct 2026 00:01:00 GMT"}, now=now))
        self.assertEqual(2.5, retry_after({"retry-after-ms": "2500"}))
        for value in ("NaN", "Infinity", "wrong", None):
            self.assertIsNone(retry_after({"Retry-After": value}))

    def test_shared_429_waits_at_least_60_and_resumes_gradually(self):
        control, clock = self.controller()
        first = self.admit(control, clock)
        control.complete(first, error=RateLimitError(), cooldown=60)
        new = self.admit(control, clock)
        self.assertEqual(60, new.wait_seconds)
        self.assertIn("rate_limit_cooldown", new.wait_reasons)
        retry = self.admit(control, clock)
        self.assertGreater(retry.wait_seconds, 0)
        self.assertEqual(60.25, clock.value)

    def test_retry_after_beyond_deadline_aborts_without_short_wait_or_send(self):
        control, clock = self.controller()
        first = self.admit(control, clock)
        control.complete(first, error=RateLimitError(), cooldown=60)
        with self.assertRaises(RequestDeadlineError):
            self.admit(control, clock, deadline=30)
        self.assertEqual([], clock.waits)
        self.assertEqual(1, control._sequence)

    def test_timeout_is_individual_and_other_provider_ignores_cooldown(self):
        openai, clock = self.controller()
        first = self.admit(openai, clock)
        openai.complete(first, error=ServiceTimeoutError(), cooldown=60)
        self.assertEqual(0, self.admit(openai, clock).wait_seconds)
        openai.complete(None, error=RateLimitError(), cooldown=60)
        jev, jev_time = self.controller()
        self.assertEqual(0, self.admit(jev, jev_time).wait_seconds)

    def test_repeated_5xx_pauses_only_after_threshold_then_ramps(self):
        control, clock = self.controller(server_error_threshold=3, server_error_cooldown=7)
        for index in range(3):
            admission = self.admit(control, clock)
            self.assertEqual(0, admission.wait_seconds)
            control.complete(admission, error=ServiceUnavailableError(), cooldown=1)
        admission = self.admit(control, clock)
        self.assertEqual(7, admission.wait_seconds)
        self.assertIn("service_error_cooldown", admission.wait_reasons)
        self.assertGreater(self.admit(control, clock).wait_seconds, 0)

    def test_authentication_and_quota_stop_future_requests_and_retries(self):
        for error in (AuthenticationError("sanitized"), ProviderConfigurationError("sanitized")):
            control, clock = self.controller()
            admission = self.admit(control, clock)
            control.complete(admission, error=error)
            with self.assertRaises(type(error)):
                self.admit(control, clock)
            self.assertEqual(1, control._sequence)
            self.assertEqual([], clock.waits)

    def test_request_and_token_windows_apply_to_failed_requests_and_retries(self):
        control, clock = self.controller(requests=2, tokens=100, window_seconds=1)
        first = self.admit(control, clock, 60)
        control.complete(first, error=ServiceTimeoutError())
        retry = self.admit(control, clock, 60)
        self.assertEqual(1, retry.wait_seconds)
        self.assertIn("token_window", retry.wait_reasons)
        control.complete(retry)
        control2, clock2 = self.controller(requests=2, window_seconds=1)
        self.admit(control2, clock2)
        self.assertEqual(.5, self.admit(control2, clock2).wait_seconds)
        self.assertEqual(.5, self.admit(control2, clock2).wait_seconds)

    def test_oversized_single_payload_is_rejected_without_waiting(self):
        control, clock = self.controller(tokens=100)
        with self.assertRaises(ConfigurationError):
            self.admit(control, clock, 101)
        self.assertEqual([], clock.waits)
        self.assertEqual(0, control._sequence)

    def test_cancelled_wait_does_not_admit_and_does_not_cancel_inflight(self):
        control, clock = self.controller(requests=1)
        in_flight = self.admit(control, clock)
        event = threading.Event()
        def cancel(seconds):
            event.set()
        with self.assertRaises(RequestCancelledError):
            control.admit(100, deadline=1000, sleep=cancel, cancel_event=event)
        control.complete(in_flight)
        self.assertEqual(1, control._sequence)

    def test_header_discovery_tightens_explicit_limits_and_accounts_inflight(self):
        control, clock = self.controller(requests=5, tokens=1000, learn_headers=True)
        first = self.admit(control, clock, 100)
        # A second in-flight send can predate a later discovery/update.
        control._bootstrap_done = True
        second = self.admit(control, clock, 100)
        control.complete(first, headers={"x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "1000", "x-ratelimit-remaining-tokens": "150",
            "x-ratelimit-reset-tokens": "1m", "x-ratelimit-remaining-requests": "2",
            "x-ratelimit-reset-requests": "1s"})
        self.assertEqual(5, control.snapshot()["request_limit"])
        self.assertEqual(900, control.snapshot()["token_limit"])
        self.assertEqual(50, control._header_tokens)
        self.assertEqual(1, control._header_requests)
        control.complete(second)
        third = self.admit(control, clock, 100)
        self.assertGreaterEqual(third.wait_seconds, 60)
        self.assertIn("header_token_remaining", third.wait_reasons)

    def test_invalid_header_values_do_not_establish_rates(self):
        control, clock = self.controller(learn_headers=True)
        first = self.admit(control, clock)
        control.complete(first, headers={"x-ratelimit-limit-requests": "NaN",
                                       "x-ratelimit-limit-tokens": "-5"})
        self.assertIsNone(control.snapshot()["request_limit"])
        self.assertIsNone(control.snapshot()["token_limit"])
        from open_video_summary.utils.request_control import _reset_seconds
        self.assertIsNone(_reset_seconds("9" * 500 + "s"))

    def test_explicit_scope_groups_credential_project_and_model_family(self):
        scope = RequestScope()
        config = LLMConfig(provider="openai", api_key="private", limit_group="family")
        limits = RequestLimits()
        one = scope.controller(config, limits)
        self.assertIs(one, scope.controller(replace(config, model="another-model"), limits))
        for changed in (replace(config, project="other"), replace(config, api_key="different"),
                        replace(config, provider="typesafe"), replace(config, limit_group="other")):
            self.assertIsNot(one, scope.controller(changed, limits))
        self.assertNotIn("private", repr(one.snapshot()))


class AdapterAdmissionTests(unittest.TestCase):
    def openai(self, outputs, **config):
        clock = FakeTime()
        create = Mock(side_effect=outputs)
        adapter = OpenAIAdapter(config=LLMConfig(provider="openai", learn_rate_limits=False, **config),
            client=SimpleNamespace(responses=SimpleNamespace(create=create)), sleep=clock.sleep,
            jitter=lambda low, high: 0)
        adapter.controller = RequestController(adapter.controller.limits, clock=clock.now)
        return adapter, create, clock

    def test_openai_full_payload_output_reservation_every_physical_retry(self):
        error = ExternalStatusError(429)
        error.response = SimpleNamespace(headers={"Retry-After": "60"})
        adapter, create, clock = self.openai([error, response('{"items":["ok"]}')],
            max_attempts=2, max_output_tokens=100, token_limit=10000)
        result = adapter.generate(GenerationRequest("state and questions", OutputSpec(kind="string_list")))
        self.assertEqual(2, create.call_count)
        self.assertEqual(60, clock.value)
        self.assertEqual(2, adapter.controller._sequence)
        self.assertTrue(all(item.request_sent for item in adapter.records))
        self.assertEqual(result.metadata.estimated_input_tokens + 100, result.metadata.reserved_tokens)
        self.assertEqual(100, create.call_args.kwargs["max_output_tokens"])
        self.assertGreater(result.metadata.estimated_input_tokens, len("state and questions"))
        self.assertIn("rate_limit_cooldown", result.metadata.wait_reasons)

    def test_openai_long_header_exceeds_deadline_no_second_physical_send(self):
        error = ExternalStatusError(429)
        error.response = SimpleNamespace(headers={"Retry-After": "60"})
        adapter, create, clock = self.openai([error, response("ok")], operation_timeout_seconds=30)
        with self.assertRaises(RequestDeadlineError):
            adapter.generate(GenerationRequest("text"))
        self.assertEqual(1, create.call_count)
        self.assertEqual([], clock.waits)
        self.assertFalse(adapter.records[-1].request_sent)

    def test_output_cap_incomplete_response_is_visible_and_not_accepted(self):
        adapter, create, clock = self.openai([response("partial", status="incomplete")], max_attempts=1)
        from open_video_summary.errors import InvalidResponseError
        with self.assertRaises(InvalidResponseError):
            adapter.generate(GenerationRequest("text"))
        self.assertEqual("InvalidResponseError", adapter.records[0].status)

    def test_typesafe_questions_and_criteria_are_in_input_reservation(self):
        good = TransportResponse(200, {}, b'{"answers":{"q":{"type":"noul","noul":0.9}}}')
        transport = Mock(return_value=good)
        evaluator = TypeSafeEvaluator(TypeSafeConfig(api_key="offline"), transport=transport)
        evaluator.evaluate("state", noul={"q": "A long question " * 10})
        payload = transport.call_args.kwargs["body"]
        metadata = evaluator.records[0]
        self.assertEqual(estimate_input_tokens(payload), metadata.reserved_tokens)
        self.assertGreater(metadata.reserved_tokens, len("state") + 100)
        self.assertEqual(60, metadata.request_limit)
        self.assertEqual(80000, metadata.token_limit)

    def test_typesafe_oversized_payload_prevents_transport_and_billing_is_terminal(self):
        transport = Mock()
        evaluator = TypeSafeEvaluator(TypeSafeConfig(api_key="offline", token_limit=100), transport=transport)
        with self.assertRaises(ConfigurationError):
            evaluator.evaluate("state", noul={"q": "Question"})
        transport.assert_not_called()
        self.assertFalse(evaluator.records[0].request_sent)
        transport = Mock(return_value=TransportResponse(429, {}, b'{"error":{"code":"insufficient_quota"}}'))
        evaluator = TypeSafeEvaluator(TypeSafeConfig(api_key="offline"), transport=transport)
        for index in range(2):
            with self.assertRaises(ProviderConfigurationError):
                evaluator.evaluate("state", noul={"q": "Question"})
        self.assertEqual(1, transport.call_count)

    def test_unknown_headers_slow_first_timeout_does_not_hold_other_worker(self):
        scope, first_started, second_finished = RequestScope(), threading.Event(), threading.Event()
        def slow(**kwargs):
            first_started.set()
            if not second_finished.wait(2):
                raise AssertionError("Other worker was blocked by the first request")
            raise TimeoutError("offline")
        def fast(**kwargs):
            second_finished.set()
            return response("ok")
        config = LLMConfig(provider="openai", api_key="offline", max_attempts=1)
        one = OpenAIAdapter(config=config, request_scope=scope,
            client=SimpleNamespace(responses=SimpleNamespace(create=slow)))
        two = OpenAIAdapter(config=config, request_scope=scope,
            client=SimpleNamespace(responses=SimpleNamespace(create=fast)))
        errors = []
        def first():
            try:
                one.generate(GenerationRequest("slow"))
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=first)
        worker.start()
        self.assertTrue(first_started.wait(1))
        self.assertEqual("ok", two.generate(GenerationRequest("fast")).value)
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(errors[0], ServiceTimeoutError)
        self.assertTrue(second_finished.is_set())
        self.assertLess(two.records[0].wait_seconds, 1)

    def test_terminal_feedback_wakes_queued_admission_without_waiting_cooldown(self):
        control = RequestController(RequestLimits())
        admission = control.admit(100, deadline=control.deadline(5))
        control.complete(admission, error=RateLimitError(), cooldown=4)
        queued, failures = threading.Event(), []
        def waiting():
            queued.set()
            try:
                control.admit(100, deadline=control.deadline(5))
            except Exception as error:
                failures.append(error)
        worker = threading.Thread(target=waiting)
        worker.start()
        self.assertTrue(queued.wait(1))
        control.complete(None, error=AuthenticationError("sanitized"))
        worker.join(.5)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(failures[0], AuthenticationError)
        self.assertEqual(1, control._sequence)

    def test_openai_current_financial_codes_and_types_are_terminal(self):
        for code in ("credit_balance_exhausted", "organization_spend_limit_exceeded",
                     "project_spend_limit_exceeded", "organization_usage_limit_exceeded"):
            for field in ("code", "type"):
                with self.subTest(code=code, field=field):
                    error = ExternalStatusError(429, body={"error": {field: code}})
                    adapter, create, clock = self.openai([error, response("ok")])
                    with self.assertRaises(ProviderConfigurationError):
                        adapter.generate(GenerationRequest("text"))
                    self.assertEqual(1, create.call_count)
                    self.assertEqual([], clock.waits)

    def test_deadline_reduces_each_physical_attempt_timeout(self):
        adapter, create, clock = self.openai([], max_attempts=2, timeout_seconds=120,
            operation_timeout_seconds=50, retry_backoff_seconds=0)
        def create_response(**kwargs):
            if clock.value == 0:
                clock.value = 40
                raise TimeoutError("offline")
            return response("ok")
        create.side_effect = create_response
        adapter.generate(GenerationRequest("text"))
        self.assertEqual([50, 10], [call.kwargs["timeout"] for call in create.call_args_list])

    def test_optional_project_headers_tighten_group_token_admission(self):
        control = RequestController(RequestLimits(tokens=1000, learn_headers=True))
        admission = control.admit(100, deadline=control.deadline(10))
        control.complete(admission, headers={"x-ratelimit-limit-tokens": "1000",
            "x-ratelimit-limit-project-tokens": "500",
            "x-ratelimit-remaining-project-tokens": "200",
            "x-ratelimit-reset-project-tokens": "2s"})
        self.assertEqual(450, control.snapshot()["token_limit"])
        self.assertEqual(200, control._header_tokens)

    def test_header_learning_requires_minute_window_but_explicit_custom_window_works(self):
        with self.assertRaises(ConfigurationError):
            OpenAIAdapter(config=LLMConfig(provider="openai", rate_window_seconds=1))
        adapter = OpenAIAdapter(config=LLMConfig(provider="openai", rate_window_seconds=1,
            learn_rate_limits=False, request_limit=4))
        self.assertEqual(1, adapter.controller.snapshot()["window_seconds"])


if __name__ == "__main__":
    unittest.main()
