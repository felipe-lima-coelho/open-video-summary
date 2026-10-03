import sys
import json
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_video_summary.adapters.llm import (
    DomainResponseInterpreter,
    OllamaAdapter,
    OpenAIAdapter,
)
from open_video_summary.contracts import GenerationRequest, OutputSpec
from open_video_summary.errors import (
    AuthenticationError,
    ConfigurationError,
    InvalidResponseError,
    RateLimitError,
    ServiceTimeoutError,
    ServiceUnavailableError,
)
from open_video_summary.utils.providers import LLMConfig


def response(text, *, effort=None, status="completed", model="gpt-6-luna-snapshot"):
    return {
        "status": status,
        "model": model,
        "reasoning": {"effort": effort},
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ],
    }


class ExternalStatusError(Exception):
    def __init__(self, status, *, body=None):
        super().__init__("external body might contain credentials or transcript")
        self.status_code = status
        self.body = body


class LLMAdapterTests(unittest.TestCase):
    def openai(self, outputs, *, effort=None, model="gpt-6-luna", attempts=3):
        config = LLMConfig(
            provider="openai",
            model=model,
            base_url="https://api.openai.com/v1",
            reasoning_effort=effort,
            max_attempts=attempts,
            retry_backoff_seconds=0,
        )
        create = Mock(side_effect=outputs)
        adapter = OpenAIAdapter(
            config=config,
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            sleep=Mock(),
        )
        return adapter, create

    def test_responses_reasoning_is_omitted_when_absent(self):
        adapter, create = self.openai([response('{"0":"Ciência"}')])
        result = adapter.generate(
            GenerationRequest("topics", OutputSpec(kind="topics"))
        )
        kwargs = create.call_args.kwargs
        self.assertNotIn("reasoning", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertFalse(kwargs["store"])
        self.assertEqual("json_schema", kwargs["text"]["format"]["type"])
        self.assertTrue(kwargs["text"]["format"]["strict"])
        self.assertEqual({"0": "Ciência"}, result.value)
        self.assertIsNone(result.metadata.requested_reasoning_effort)
        self.assertIsNone(result.metadata.sent_reasoning_effort)
        self.assertIsNone(result.metadata.reported_reasoning_effort)

    def test_explicit_none_and_high_are_translated_without_temperature(self):
        for effort in ("none", "high", "max"):
            with self.subTest(effort=effort):
                adapter, create = self.openai([response("Fixed text.")], effort=effort)
                result = adapter.generate(GenerationRequest("fix"))
                self.assertEqual(
                    {"effort": effort}, create.call_args.kwargs["reasoning"]
                )
                self.assertNotIn("temperature", create.call_args.kwargs)
                self.assertEqual(effort, result.metadata.sent_reasoning_effort)
                self.assertIsNone(result.metadata.reported_reasoning_effort)

    def test_requested_and_reported_model_effort_are_separate(self):
        adapter, _ = self.openai([response("text", effort="medium")], effort="high")
        result = adapter.generate(GenerationRequest("fix"))
        self.assertEqual("gpt-6-luna", result.metadata.requested_model)
        self.assertEqual("gpt-6-luna-snapshot", result.metadata.returned_model)
        self.assertEqual("high", result.metadata.requested_reasoning_effort)
        self.assertEqual("medium", result.metadata.reported_reasoning_effort)

    def test_new_model_names_are_forwarded(self):
        adapter, create = self.openai(
            [response("text")], model="future-provider-model", effort="high"
        )
        adapter.generate(GenerationRequest("fix"))
        self.assertEqual("future-provider-model", create.call_args.kwargs["model"])

    def test_known_incompatible_efforts_fail_before_requests(self):
        for model, effort in (
            ("gpt-4o", "high"),
            ("gpt-4o-2024-08-06", "high"),
            ("gpt-4o-mini", "high"),
            ("gpt-4o-mini-2024-07-18", "high"),
            ("gpt-4.1", "high"),
            ("gpt-4.1-2025-04-14", "high"),
            ("gpt-4.1-mini", "high"),
            ("gpt-4.1-mini-2025-04-14", "high"),
            ("gpt-4.1-nano", "high"),
            ("gpt-4.1-nano-2025-04-14", "high"),
            ("gpt-3.5-turbo", "high"),
            ("gpt-3.5-turbo-0125", "high"),
            ("gpt-3.5-turbo-1106", "high"),
            ("gpt-3.5-turbo-instruct", "high"),
            ("gpt-6-luna", "minimal"),
            ("gpt-6-astra", "none"),
            ("gpt-6-astra", "minimal"),
            ("gpt-6.1-sol", "minimal"),
            ("gpt-6.1-sol", "none"),
            ("gpt-5", "none"),
            ("gpt-5", "xhigh"),
            ("gpt-5", "max"),
            ("gpt-5-2025-08-07", "none"),
            ("o3", "none"),
            ("gpt-6-luna", "invalid"),
        ):
            with self.subTest(model=model, effort=effort):
                with self.assertRaises(ConfigurationError):
                    self.openai([], model=model, effort=effort)
        with self.assertRaises(ConfigurationError):
            OllamaAdapter(config=LLMConfig(reasoning_effort="none"))

    def test_known_valid_efforts_and_unknown_variants_are_preserved(self):
        for model, effort in (
            ("gpt-6-astra", "max"),
            ("gpt-6.1-sol", "low"),
            ("gpt-5", "minimal"),
            ("gpt-5-2025-08-07", "high"),
            ("gpt-4.10", "high"),
            ("gpt-4o-future-variant", "high"),
            ("gpt-3.5-future-variant", "high"),
            ("gpt-5-future-variant", "none"),
            ("gpt-6-astra-future-variant", "none"),
            ("o10-future-model", "max"),
        ):
            with self.subTest(model=model, effort=effort):
                adapter, create = self.openai(
                    [response("text")], model=model, effort=effort
                )
                adapter.generate(GenerationRequest("synthetic request"))
                self.assertEqual(model, create.call_args.kwargs["model"])
                self.assertEqual(
                    {"effort": effort}, create.call_args.kwargs["reasoning"]
                )

    def test_duplicate_literal_identifiers_consume_finite_attempts(self):
        adapter, create = self.openai(
            [
                response("{'0':'first','0':'second'}"),
                response("{'0':'first','\\u0030':'second'}"),
                response("{'0':'valid'}"),
            ]
        )
        result = adapter.generate(GenerationRequest("topics", OutputSpec("topics")))
        self.assertEqual({"0": "valid"}, result.value)
        self.assertEqual(3, create.call_count)
        self.assertEqual(
            ["InvalidResponseError", "InvalidResponseError", "completed"],
            [record.status for record in adapter.records],
        )

    def test_openai_json_envelopes_translate_to_existing_domain_shapes(self):
        adapter, create = self.openai(
            [
                response('{"topics":[{"id":"0","label":"Tema"}]}'),
                response('{"id":"0","label":"Tema"}'),
                response('{"items":["Tema geral"]}'),
            ]
        )
        self.assertEqual(
            {"0": "Tema"},
            adapter.generate(GenerationRequest("topics", OutputSpec("topics"))).value,
        )
        self.assertEqual(
            {"0": "Tema"},
            adapter.generate(
                GenerationRequest("classify", OutputSpec("topic", ("0",)))
            ).value,
        )
        schema = create.call_args.kwargs["text"]["format"]["schema"]
        self.assertEqual(["0"], schema["properties"]["id"]["enum"])
        self.assertEqual(
            ["Tema geral"],
            adapter.generate(
                GenerationRequest("global", OutputSpec("string_list"))
            ).value,
        )

    def test_invalid_topics_and_ids_consume_the_same_finite_budget(self):
        adapter, create = self.openai(
            [
                response('{"id":"missing","label":"bad"}'),
                response('{"0":"Tema","1":"Other"}'),
                response('{"0":"Tema"}'),
            ]
        )
        result = adapter.generate(
            GenerationRequest("classify", OutputSpec("topic", ("0",)))
        )
        self.assertEqual(3, create.call_count)
        self.assertEqual(3, result.metadata.attempts)
        self.assertEqual(
            ["InvalidResponseError", "InvalidResponseError", "completed"],
            [r.status for r in adapter.records],
        )

    def test_empty_malformed_and_incomplete_responses_exhaust_attempts(self):
        adapter, create = self.openai(
            [response(""), response("{bad}"), response("{}", status="incomplete")]
        )
        with self.assertRaises(InvalidResponseError):
            adapter.generate(GenerationRequest("topics", OutputSpec("topics")))
        self.assertEqual(3, create.call_count)
        self.assertEqual(3, len(adapter.records))

    def test_refusal_is_an_internal_failure_with_no_content_in_error(self):
        refusal = response("text")
        refusal["output"][1]["content"] = [
            {"type": "refusal", "refusal": "private research text"}
        ]
        adapter, create = self.openai([refusal, refusal], attempts=2)
        with self.assertRaises(InvalidResponseError) as caught:
            adapter.generate(GenerationRequest("fix"))
        self.assertEqual(2, create.call_count)
        self.assertNotIn("private research", str(caught.exception))

    def test_timeout_and_invalid_response_share_attempt_budget(self):
        adapter, create = self.openai(
            [TimeoutError("raw message"), response("{}"), TimeoutError("raw message")]
        )
        with self.assertRaises(ServiceTimeoutError):
            adapter.generate(GenerationRequest("topics", OutputSpec("topics")))
        self.assertEqual(3, create.call_count)
        self.assertEqual(2, adapter._sleep.call_count)

    def test_status_errors_are_translated_and_terminal_errors_do_not_retry(self):
        for status, expected, calls in (
            (401, AuthenticationError, 1),
            (403, AuthenticationError, 1),
            (400, ConfigurationError, 1),
            (404, ConfigurationError, 1),
            (429, RateLimitError, 2),
            (503, ServiceUnavailableError, 2),
        ):
            with self.subTest(status=status):
                error = ExternalStatusError(
                    status, body={"error": {"param": "reasoning.effort"}}
                )
                adapter, create = self.openai([error, error], attempts=2)
                with self.assertRaises(expected) as caught:
                    adapter.generate(GenerationRequest("fix"))
                self.assertEqual(calls, create.call_count)
                self.assertNotIn("credentials", str(caught.exception))

    def test_quota_errors_are_terminal(self):
        error = ExternalStatusError(429, body={"code": "insufficient_quota"})
        adapter, create = self.openai([error, error])
        with self.assertRaises(ConfigurationError):
            adapter.generate(GenerationRequest("fix"))
        self.assertEqual(1, create.call_count)

    def test_clients_disable_sdk_retries_and_use_configured_endpoint(self):
        constructor = Mock()
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=constructor)}):
            config = LLMConfig(
                provider="openai",
                model="custom",
                base_url="https://custom.test/v1",
                api_key="test-secret",
                timeout_seconds=11,
            )
            adapter = OpenAIAdapter(config=config)
            constructor.assert_not_called()
            adapter.preflight()
            constructor.assert_called_once_with(
                api_key="test-secret",
                base_url="https://custom.test/v1",
                timeout=11,
                max_retries=0,
            )
            adapter.close()

    def test_unused_sdk_is_not_imported_and_missing_key_fails_clearly(self):
        with patch.dict(
            sys.modules, {"openai": None, "ollama": None, "whisper_timestamped": None}
        ):
            OllamaAdapter()
            remote = OpenAIAdapter()
            with self.assertRaisesRegex(ConfigurationError, "OPENAI_API_KEY"):
                remote.preflight()
            with self.assertRaisesRegex(ConfigurationError, "ollama package"):
                OllamaAdapter().preflight()

    def test_ollama_translation_and_legacy_pattern_api(self):
        client = SimpleNamespace(
            generate=Mock(
                return_value={
                    "response": 'prefix {"0":"Tema"} suffix',
                    "model": "gemma2:latest",
                }
            )
        )
        adapter = OllamaAdapter(client=client)
        self.assertEqual(
            '{"0":"Tema"}',
            adapter.generate_pattern(
                "prompt",
                r"(\{.*?\})",
                options={"format": "json", "temperature": 0.2},
            ),
        )
        client.generate.assert_called_once_with(
            model="gemma2",
            prompt="prompt",
            stream=False,
            format="json",
            options={"temperature": 0.2},
        )

    def test_legacy_pattern_mismatch_consumes_attempts(self):
        client = SimpleNamespace(generate=Mock(return_value={"response": "no match"}))
        adapter = OllamaAdapter(client=client, max_attempts=2, sleep=Mock())
        with self.assertRaises(InvalidResponseError):
            adapter.generate_pattern("prompt", r"(\{.*?\})")
        self.assertEqual(2, client.generate.call_count)

    def test_ollama_preflight_uses_selected_server_model(self):
        client = SimpleNamespace(
            list=Mock(return_value={"models": [{"name": "gemma2:latest"}]})
        )
        OllamaAdapter(client=client).preflight()
        with self.assertRaisesRegex(ConfigurationError, "missing"):
            OllamaAdapter(model="missing", client=client).preflight()

    def test_image_inputs_are_translated_within_adapter(self):
        adapter, create = self.openai([response("text")])
        adapter.generate(GenerationRequest("describe", images=(b"jpeg-data",)))
        content = create.call_args.kwargs["input"][0]["content"]
        self.assertEqual("input_text", content[0]["type"])
        self.assertTrue(content[1]["image_url"].startswith("data:image/jpeg;base64,"))


class ResponseInterpreterTests(unittest.TestCase):
    def test_structure_validation(self):
        parser = DomainResponseInterpreter()
        for text, spec in (
            ("{}", OutputSpec("topics")),
            ('{"0":""}', OutputSpec("topics")),
            ('{"0":true}', OutputSpec("topics")),
            ("[]", OutputSpec("string_list")),
            ("[1]", OutputSpec("string_list")),
            ('{"0":"a","1":"b"}', OutputSpec("topics", max_items=1)),
            (
                '{"topics":[{"id":"0","label":"a"},{"id":"0","label":"b"}]}',
                OutputSpec("topics"),
            ),
            ('{"0":"a","0":"b"}', OutputSpec("topics")),
        ):
            with self.subTest(text=text):
                with self.assertRaises(InvalidResponseError):
                    parser.interpret(text, spec)

    def test_fenced_json_and_existing_python_literals(self):
        parser = DomainResponseInterpreter()
        self.assertEqual(
            {"0": "Tema"},
            parser.interpret('```json\n{"0":"Tema"}\n```', OutputSpec("topics")),
        )
        self.assertEqual(
            {"0": "Tema"}, parser.interpret("{'0': 'Tema'}", OutputSpec("topics"))
        )

    def test_boolean_answers_are_normalized_and_validated(self):
        parser = DomainResponseInterpreter()
        spec = OutputSpec(kind="answers", answer_ids=("0", "1"))
        self.assertEqual(
            {"0": True, "1": None}, parser.interpret("{'0': True, '1': None}", spec)
        )
        self.assertEqual(
            {"0": False, "1": True},
            parser.interpret(
                '{"answers":[{"id":"0","answer":false},{"id":"1","answer":true}]}',
                spec,
            ),
        )
        for text in ('{"0":true}', '{"0":"true","1":false}', '{"9":true,"1":false}'):
            with self.assertRaises(InvalidResponseError):
                parser.interpret(text, spec)

    def test_literal_dictionary_keys_are_checked_before_evaluation(self):
        parser = DomainResponseInterpreter()
        for text in (
            "{'0':'a','0':'b'}",
            "{'0':'a','\\u0030':'b'}",
            "{'topics':[{'id':'0','id':'1','label':'a'}]}",
            "{True:'a',1:'b'}",
            "{[0]:'a'}",
            "{{'0':'a'}:'b'}",
        ):
            with self.subTest(text=text):
                with self.assertRaises(InvalidResponseError):
                    parser.interpret(text, OutputSpec("topics"))


@unittest.skipUnless(
    importlib.util.find_spec("httpx2"), "Pinned OpenAI SDK transport is not installed."
)
class InstalledOpenAISDKTests(unittest.TestCase):
    def test_actual_sdk_serializes_responses_request_on_offline_transport(self):
        import httpx2
        from openai import OpenAI

        sent = []

        def handle(request):
            sent.append(json.loads(request.content))
            fixture = response('{"topics":[{"id":"0","label":"Tema"}]}', effort="high")
            fixture.update(
                {"id": "resp_offline", "object": "response", "created_at": 0}
            )
            return httpx2.Response(200, json=fixture)

        with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
            with OpenAI(
                api_key="synthetic-offline-key",
                base_url="https://offline.example/v1",
                max_retries=0,
                timeout=2,
                http_client=transport,
            ) as client:
                config = LLMConfig(
                    provider="openai", model="gpt-6-luna", reasoning_effort="high"
                )
                adapter = OpenAIAdapter(config=config, client=client)
                result = adapter.generate(
                    GenerationRequest("synthetic topics", OutputSpec("topics"))
                )
        self.assertEqual({"0": "Tema"}, result.value)
        self.assertEqual("high", result.metadata.reported_reasoning_effort)
        self.assertEqual({"effort": "high"}, sent[0]["reasoning"])
        self.assertEqual("json_schema", sent[0]["text"]["format"]["type"])
        self.assertNotIn("temperature", sent[0])

    def test_actual_sdk_status_error_obeys_only_application_budget(self):
        import httpx2
        from openai import OpenAI

        sent = []

        def handle(request):
            sent.append(request)
            return httpx2.Response(
                429,
                json={
                    "error": {
                        "message": "synthetic limit",
                        "type": "rate_limit_error",
                        "code": "rate_limit",
                    }
                },
            )

        with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
            with OpenAI(
                api_key="synthetic-offline-key", max_retries=0, http_client=transport
            ) as client:
                adapter = OpenAIAdapter(
                    config=LLMConfig(
                        provider="openai", model="future-model", max_attempts=2
                    ),
                    client=client,
                    sleep=Mock(),
                )
                with self.assertRaises(RateLimitError):
                    adapter.generate(GenerationRequest("synthetic prompt"))
        self.assertEqual(2, len(sent))

    def test_actual_sdk_timeout_is_translated(self):
        import httpx2
        from openai import OpenAI

        def handle(request):
            raise httpx2.ReadTimeout("synthetic timeout", request=request)

        with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
            with OpenAI(
                api_key="synthetic-offline-key", max_retries=0, http_client=transport
            ) as client:
                adapter = OpenAIAdapter(
                    config=LLMConfig(
                        provider="openai", model="future-model", max_attempts=1
                    ),
                    client=client,
                    sleep=Mock(),
                )
                with self.assertRaises(ServiceTimeoutError):
                    adapter.generate(GenerationRequest("synthetic prompt"))


if __name__ == "__main__":
    unittest.main()
