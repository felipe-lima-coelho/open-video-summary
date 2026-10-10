"""Offline adapter and orchestration checks for timeout-only focus recovery."""

import hashlib
import importlib.util
import json
import threading
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_information_analysis import candidate, videos
from test_information_inventory import InventoryEvaluator
from test_llm_adapters import ExternalStatusError, response
from test_request_control import FakeTime
from open_video_summary.adapters.information_schema import information_schema
from open_video_summary.adapters.llm import OpenAIAdapter
from open_video_summary.contracts import GenerationAttemptMetadata, GenerationRequest, GenerationRetryPlan, ServiceMetadata
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer, PROTOCOL, _AnalysisRun
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import capture_snapshot, fingerprint
from open_video_summary.core.summarizers.information_discovery import DiscoveryLookahead
from open_video_summary.core.summarizers.information_inventory import FOCUS_INSTRUCTION
from open_video_summary.errors import AuthenticationError, ConfigurationError, RequestCancelledError, RequestDeadlineError, ServiceTimeoutError
from open_video_summary.utils.providers import LLMConfig
from open_video_summary.utils.request_control import RequestController, RequestLimits


class ReadTimeout(TimeoutError):
    pass


class FocusRetryTests(unittest.TestCase):
    def setUp(self):
        self.body = {"candidates": [], "issues": []}
        self.config = LLMConfig(provider="openai", model="gpt-6-luna", reasoning_effort="high",
            retry_backoff_seconds=0, learn_rate_limits=False)

    def adapter(self, outputs=(), **settings):
        create = Mock(side_effect=outputs)
        adapter = OpenAIAdapter(config=replace(self.config, **settings),
            client=SimpleNamespace(responses=SimpleNamespace(create=create)), sleep=Mock(), jitter=lambda *_: 0)
        return adapter, create

    def run_for(self, adapter, source="O prazo é 30 dias."):
        run = _AnalysisRun(InformationAnalyzer(adapter, InventoryEvaluator(), InformationAnalysisConfig(
            concurrency=1, qa_enabled=False, max_coverage_rounds=0, max_granularity_checks=0)),
            capture_snapshot(videos([source])))
        run.active_target = "v0:s0"
        return run

    def requests(self, adapter):
        run = self.run_for(adapter)
        target = run.segments["v0:s0"]
        prompt, spec = run._inventory_request(target, "discover_coverage_foci", FOCUS_INSTRUCTION)
        return run, GenerationRequest(prompt, spec, temperature=0.0), run._focus_timeout_fallback(target)

    def success(self, body=None):
        return response(json.dumps(body or self.body), effort="high")

    def formats(self, create):
        return [call.kwargs["text"]["format"]["name"] for call in create.call_args_list]

    def test_healthy_primary_is_original_prompt_schema_and_full_provider_payload(self):
        adapter, create = self.adapter([self.success(), self.success()])
        run, primary, fallback = self.requests(adapter)
        legacy_instruction = (
            "Independently traverse the ORIGINAL target, including content without numbers or "
            "other marked words. Propose each distinct expressed content as one contextualized "
            "source focus, with literal evidence and every necessary qualifier. These are "
            "coverage hypotheses, not accepted inventory units. Do not infer coverage from "
            "citations. Do not turn conditions, attribution or modality into asserted events. "
            "No existing inventory is supplied. Set question and answer to null.")
        self.assertEqual(legacy_instruction, FOCUS_INSTRUCTION)
        self.assertEqual("information_units", primary.output.kind)
        self.assertTrue(primary.prompt.startswith(PROTOCOL + "\n" + legacy_instruction + "\n"))
        self.assertEqual(primary.prompt.split("\nInput:\n")[1], fallback.prompt.split("\nInput:\n")[1])
        plain = adapter.generate(primary)
        result = adapter.generate_with_timeout_fallback(primary, fallback)
        first, second = [dict(call.kwargs) for call in create.call_args_list]
        first.pop("timeout"); second.pop("timeout")
        self.assertEqual(first, second)
        self.assertIs(type(plain.metadata), ServiceMetadata)
        self.assertIsInstance(result.metadata, GenerationAttemptMetadata)
        self.assertEqual("primary", result.metadata.request_variant)
        self.assertEqual(fingerprint(asdict(primary)), result.metadata.request_fingerprint)
        self.assertEqual(hashlib.sha256(primary.prompt.encode()).hexdigest(), result.metadata.prompt_fingerprint)
        self.assertEqual(fingerprint(information_schema(primary.output)), result.metadata.output_schema_fingerprint)
        with self.assertRaises(FrozenInstanceError):
            result.metadata.retry_plan.primary = fallback

    def test_sent_read_timeout_switches_once_and_retains_failure_and_plan(self):
        adapter, create = self.adapter([ReadTimeout("private"), ReadTimeout("private"), self.success()])
        _, primary, fallback = self.requests(adapter)
        result = adapter.generate_with_timeout_fallback(primary, fallback)
        self.assertEqual(["ovs_information_units", "ovs_information_foci", "ovs_information_foci"], self.formats(create))
        self.assertEqual(["primary", "timeout_fallback", "timeout_fallback"], [r.request_variant for r in adapter.records])
        self.assertEqual(["read", "read", None], [r.timeout_phase for r in adapter.records])
        self.assertEqual(["ServiceTimeoutError", "ServiceTimeoutError", "completed"], [r.status for r in adapter.records])
        self.assertTrue(all(r.request_sent and r.retry_plan == GenerationRetryPlan(primary, fallback) for r in adapter.records))
        self.assertEqual(3, result.metadata.attempts)
        self.assertEqual(fingerprint(asdict(fallback)), result.metadata.request_fingerprint)
        self.assertEqual([False, False, True], [r.response_received for r in adapter.records])

    def test_admission_estimates_only_each_active_request_and_rate_retry_keeps_fallback(self):
        limited = ExternalStatusError(429)
        limited.response = SimpleNamespace(headers={"Retry-After": "2"})
        adapter, create = self.adapter([ReadTimeout("offline"), limited, self.success()])
        _, primary, fallback = self.requests(adapter)
        adapter.generate_with_timeout_fallback(primary, fallback)
        self.assertEqual(["ovs_information_units", "ovs_information_foci", "ovs_information_foci"], self.formats(create))
        for call, audit in zip(create.call_args_list, adapter.records):
            payload = dict(call.kwargs); payload.pop("timeout")
            expected = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()) + 64
            self.assertEqual(expected, audit.estimated_input_tokens)
            self.assertEqual(expected + 16384, audit.reserved_tokens)
        self.assertNotEqual(adapter.records[0].estimated_input_tokens, adapter.records[1].estimated_input_tokens)
        self.assertIn("rate_limit_cooldown", adapter.records[2].wait_reasons)

    def test_oversized_unused_fallback_never_blocks_healthy_primary(self):
        adapter, create = self.adapter([self.success()])
        run, primary, fallback = self.requests(adapter)
        run.config = replace(run.config, max_context_chars=len(primary.prompt))
        proposals, _ = run._inventory_generate(run.segments["v0:s0"], "discover_coverage_foci", FOCUS_INSTRUCTION)
        self.assertEqual([], proposals)
        self.assertEqual(["ovs_information_units"], self.formats(create))
        self.assertEqual(1, run.call_count)
        self.assertGreater(len(fallback.prompt), len(primary.prompt))
        self.assertTrue(any(i.kind == "focus_timeout_fallback_context_limit" for i in run.issues))

    def test_cancel_after_sent_read_timeout_keeps_failure_without_fallback(self):
        adapter, create = self.adapter()
        _, primary, fallback = self.requests(adapter)
        adapter.cancel_event = threading.Event()
        def interrupted(**kwargs):
            adapter.cancel_event.set()
            raise ReadTimeout("offline")
        create.side_effect = interrupted
        with self.assertRaises(RequestCancelledError):
            adapter.generate_with_timeout_fallback(primary, fallback)
        self.assertEqual(1, create.call_count)
        self.assertEqual("ServiceTimeoutError", adapter.records[0].status)
        self.assertEqual("primary", adapter.records[0].request_variant)

    def test_other_errors_and_non_read_timeouts_preserve_primary(self):
        errors = [ExternalStatusError(429), ExternalStatusError(503), ConnectionError("private"),
            TimeoutError("unknown"), self.success({"invalid": True})]
        errors.extend(type(name, (TimeoutError,), {})("private")
                      for name in ("ConnectTimeout", "PoolTimeout", "WriteTimeout"))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                adapter, create = self.adapter([error, self.success()])
                _, primary, fallback = self.requests(adapter)
                adapter.generate_with_timeout_fallback(primary, fallback)
                self.assertEqual(["ovs_information_units"] * 2, self.formats(create))
                self.assertEqual(["primary"] * 2, [r.request_variant for r in adapter.records])

    def test_unsent_read_timeout_does_not_arm_fallback(self):
        adapter, create = self.adapter([self.success()])
        _, primary, fallback = self.requests(adapter)
        admit = adapter._admit
        attempts = 0
        def delayed_admission(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ReadTimeout("unsent")
            return admit(*args, **kwargs)
        with patch.object(adapter, "_admit", side_effect=delayed_admission):
            adapter.generate_with_timeout_fallback(primary, fallback)
        self.assertEqual(["ovs_information_units"], self.formats(create))
        self.assertFalse(adapter.records[0].request_sent)
        self.assertEqual("primary", adapter.records[-1].request_variant)

    def test_valid_response_followed_by_local_timeout_never_switches(self):
        adapter, create = self.adapter([self.success(), self.success()])
        _, primary, fallback = self.requests(adapter)
        with patch.object(adapter.interpreter, "interpret", side_effect=[ReadTimeout("local"), self.body]):
            adapter.generate_with_timeout_fallback(primary, fallback)
        self.assertEqual(["ovs_information_units"] * 2, self.formats(create))

    def test_max_attempts_one_and_expired_deadline_do_not_send_fallback(self):
        for attempts, expired in ((1, False), (3, True)):
            with self.subTest(attempts=attempts, expired=expired):
                adapter, create = self.adapter([], max_attempts=attempts)
                _, primary, fallback = self.requests(adapter)
                clock = FakeTime()
                adapter.controller = RequestController(RequestLimits(), clock=clock.now)
                def fail(**kwargs):
                    if expired:
                        clock.value = 360
                    raise ReadTimeout("offline")
                create.side_effect = fail
                with self.assertRaises(RequestDeadlineError if expired else ServiceTimeoutError):
                    adapter.generate_with_timeout_fallback(primary, fallback)
                self.assertEqual(1, create.call_count)
                self.assertEqual("primary", adapter.records[0].request_variant)

    def test_remaining_original_deadline_reduces_fallback_attempt_timeout(self):
        adapter, create = self.adapter([], retry_backoff_seconds=1)
        _, primary, fallback = self.requests(adapter)
        clock = FakeTime()
        adapter.controller = RequestController(RequestLimits(), clock=clock.now)
        adapter._sleep = clock.sleep
        def respond(**kwargs):
            if not clock.value:
                clock.value = 340
                raise ReadTimeout("offline")
            return self.success()
        create.side_effect = respond
        adapter.generate_with_timeout_fallback(primary, fallback)
        self.assertEqual([120, 19], [call.kwargs["timeout"] for call in create.call_args_list])
        self.assertEqual(360, adapter._operation_deadline)
        self.assertEqual([1], clock.waits)

    def test_terminal_and_cancelled_operations_never_fall_back(self):
        adapter, create = self.adapter([ExternalStatusError(401), self.success()])
        _, primary, fallback = self.requests(adapter)
        with self.assertRaises(AuthenticationError):
            adapter.generate_with_timeout_fallback(primary, fallback)
        self.assertEqual(1, create.call_count)
        adapter, create = self.adapter([self.success()])
        adapter.cancel_event = threading.Event(); adapter.cancel_event.set()
        with self.assertRaises(RequestCancelledError):
            adapter.generate_with_timeout_fallback(primary, fallback)
        self.assertEqual(0, create.call_count)

    def test_other_operations_reject_explicit_fallback_and_keep_metadata_shape(self):
        adapter, create = self.adapter([ReadTimeout("offline"), self.success()])
        run, primary, fallback = self.requests(adapter)
        with self.assertRaises(ConfigurationError):
            run._generate_request("extract_direct", primary.prompt, primary.output, timeout_fallback=fallback)
        adapter.generate(primary)
        self.assertEqual(["ovs_information_units"] * 2, self.formats(create))
        self.assertTrue(all(type(r) is ServiceMetadata for r in adapter.records))

    def test_nullable_fallback_is_one_logical_call_and_semantically_validated(self):
        raw = candidate("v0:s0", "O prazo é 30 dias.", quantities=("30 dias",))
        raw["evidence"][0].update(start_char=None, end_char=None)
        adapter, _ = self.adapter([ReadTimeout("offline"), self.success({"candidates": [raw], "issues": []})])
        run = self.run_for(adapter)
        run._discover_foci(run.segments["v0:s0"])
        calls = [call for call in run.calls if call.operation == "discover_coverage_foci"]
        self.assertEqual(1, len(calls))
        audit = json.loads(calls[0].metadata_json)
        self.assertEqual(2, len(audit["attempt_records"]))
        self.assertEqual("primary", audit["attempt_records"][0]["request_variant"])
        self.assertEqual("timeout_fallback", audit["request_variant"])
        focus = run.coverage_foci[0]
        self.assertEqual("accepted", focus.proposal_record.validation)
        self.assertIsNone(json.loads(focus.proposal_record.raw_json)["evidence"][0]["start_char"])
        self.assertIsInstance(focus.proposition.evidence[0].start_char, int)

    def test_lookahead_keeps_private_adapter_same_scope_and_one_reserved_call(self):
        parent = OpenAIAdapter(config=self.config, sleep=Mock(), jitter=lambda *_: 0)
        run, primary, fallback = self.requests(parent)
        run.target_index = 0
        run.discovery = DiscoveryLookahead(2, generator=parent, evaluator=run.evaluator)
        clients, threads = {}, []
        def get_client(adapter):
            threads.append(threading.get_ident())
            if id(adapter) not in clients:
                create = Mock(side_effect=[ReadTimeout("offline"), self.success()])
                clients[id(adapter)] = (adapter, SimpleNamespace(responses=SimpleNamespace(create=create)))
            return clients[id(adapter)][1]
        try:
            with patch.object(OpenAIAdapter, "_get_client", get_client):
                run._seed_discovery(run.segments["v0:s0"], "direct", 0, None, 0)
                result = run._generate_request("discover_coverage_foci", primary.prompt, primary.output,
                    timeout_fallback=fallback)
            child = next(iter(clients.values()))[0]
            self.assertIsNot(parent, child)
            self.assertIs(parent.request_scope, child.request_scope)
            self.assertIs(parent.controller, child.controller)
            self.assertNotIn(threading.get_ident(), threads)
            self.assertEqual("timeout_fallback", result.metadata.request_variant)
            self.assertEqual(1, run.call_count)
            self.assertEqual(1, len(run.calls))
            self.assertEqual(0, run.budget.reserved)
            self.assertEqual(1, run.discovery.snapshot()["consumed"])
        finally:
            run.discovery.close()


@unittest.skipUnless(importlib.util.find_spec("httpx2"), "Pinned SDK transport is not installed.")
class InstalledTimeoutPhaseTests(unittest.TestCase):
    setUp = FocusRetryTests.setUp
    run_for = FocusRetryTests.run_for
    requests = FocusRetryTests.requests
    success = FocusRetryTests.success

    def test_actual_sdk_read_and_connect_causes_select_only_read_fallback(self):
        import httpx2
        from openai import OpenAI
        for error_type, expected in ((httpx2.ReadTimeout, "ovs_information_foci"),
                                     (httpx2.ConnectTimeout, "ovs_information_units")):
            sent = []
            def handle(request):
                sent.append(json.loads(request.content))
                if len(sent) == 1:
                    raise error_type("synthetic", request=request)
                body = self.success(); body.update(id="offline", object="response", created_at=0)
                return httpx2.Response(200, json=body)
            with self.subTest(error_type=error_type.__name__), httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
                with OpenAI(api_key="synthetic", http_client=transport) as client:
                    adapter = OpenAIAdapter(config=self.config, client=client, sleep=Mock(), jitter=lambda *_: 0)
                    _, primary, fallback = self.requests(adapter)
                    adapter.generate_with_timeout_fallback(primary, fallback)
            self.assertEqual(2, len(sent))
            self.assertEqual(expected, sent[1]["text"]["format"]["name"])
            self.assertEqual("read" if error_type is httpx2.ReadTimeout else "connect", adapter.records[0].timeout_phase)


if __name__ == "__main__":
    unittest.main()
