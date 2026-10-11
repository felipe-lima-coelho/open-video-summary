"""Offline wire and isolation checks for the information-only OpenAI profile."""

import importlib.util
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_video_summary.adapters.factory import create_llm
from open_video_summary.adapters.information_llm import InformationOpenAIAdapter
from open_video_summary.adapters.llm import OllamaAdapter, OpenAIAdapter
from open_video_summary.contracts import GenerationRequest, OutputSpec
from open_video_summary.core.summarizers.information_config import configured_information_analyzer
from open_video_summary.core.summarizers.information_reliability import analysis_manifest
from open_video_summary.errors import ConfigurationError, InvalidResponseError, ProviderConfigurationError
from open_video_summary.utils.providers import LLMConfig, load_provider_config
from test_llm_adapters import ExternalStatusError


def chat_response(text="ok", *, fingerprint="fp_fixture", finish="stop", refusal=None):
    return {"id": "chatcmpl_fixture", "object": "chat.completion", "created": 0,
            "model": "gpt-6-luna-2026-09-01", "system_fingerprint": fingerprint,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text, "refusal": refusal}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}


class InformationProviderProfileTests(unittest.TestCase):
    config = LLMConfig(provider="openai", model="gpt-6-luna", reasoning_effort="none",
        base_url="https://api.openai.com/v1", retry_backoff_seconds=0, learn_rate_limits=False)

    def adapter(self, *responses, **settings):
        create = Mock(side_effect=responses)
        return InformationOpenAIAdapter(config=replace(self.config, **settings),
            client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
            sleep=Mock()), create

    def test_profile_overrides_only_information_luna_and_preserves_shared_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "absent.env"
            for model in ("gpt-6-luna", "gpt-6-luna-2026-09-01"):
                overrides = {"llm_provider": "openai", "llm_model": model, "reasoning_effort": "high"}
                general = load_provider_config(overrides, environ={}, env_file=fixture).llm
                analyzer = configured_information_analyzer(overrides, environ={}, env_file=fixture)
                self.assertIsInstance(analyzer.generator, InformationOpenAIAdapter)
                self.assertEqual("none", analyzer.generator.config.reasoning_effort)
                self.assertEqual(replace(general, reasoning_effort="none"), analyzer.generator.config)
                self.assertEqual("high", create_llm(general).config.reasoning_effort)
                self.assertIs(type(create_llm(general)), OpenAIAdapter)
            for model in ("gpt-6-luna-other", "gpt-6-astra"):
                analyzer = configured_information_analyzer(
                    {"llm_provider": "openai", "llm_model": model, "reasoning_effort": "high"},
                    environ={}, env_file=fixture)
                self.assertIs(type(analyzer.generator), OpenAIAdapter)
                self.assertEqual("high", analyzer.generator.config.reasoning_effort)
            local = configured_information_analyzer(environ={}, env_file=fixture)
            self.assertIs(type(local.generator), OllamaAdapter)
            self.assertIsNone(local.generator.config.reasoning_effort)

    def test_fixed_settings_and_fingerprint_are_recorded_for_every_attempt(self):
        adapter, create = self.adapter(chat_response(finish="length", fingerprint="fp_first"),
                                       chat_response(fingerprint="fp_second"), chat_response(fingerprint=None))
        result = adapter.generate(GenerationRequest("information", temperature=0))
        later = adapter.generate(GenerationRequest("fresh information", temperature=0))
        self.assertEqual(["fp_first", "fp_second", None], [item.system_fingerprint for item in adapter.records])
        self.assertEqual("fp_second", result.metadata.system_fingerprint)
        self.assertIsNone(later.metadata.system_fingerprint)
        for call, record in zip(create.call_args_list, adapter.records):
            self.assertEqual((42, 0, "none"), tuple(call.kwargs[key] for key in ("seed", "temperature", "reasoning_effort")))
            self.assertEqual((42, 42, 0, "none", "chat.completions"),
                (record.requested_seed, record.seed_sent, record.temperature_sent,
                 record.sent_reasoning_effort, record.api_endpoint))
            self.assertIsNone(record.reported_reasoning_effort)
            self.assertEqual((12, 3), (record.input_tokens, record.output_tokens))

    def test_unsupported_fixed_setting_is_terminal_without_parameter_fallback(self):
        for parameter in ("seed", "temperature", "reasoning_effort"):
            adapter, create = self.adapter(ExternalStatusError(400, body={"error": {"param": parameter}}))
            with self.subTest(parameter=parameter), self.assertRaisesRegex(ProviderConfigurationError, parameter):
                adapter.generate(GenerationRequest("information", temperature=0))
            self.assertEqual(1, create.call_count)
            self.assertEqual(42, adapter.records[0].seed_sent)
            self.assertIsNone(adapter.records[0].system_fingerprint)
            adapter._sleep.assert_not_called()

    def test_known_incompatible_profile_and_changed_temperature_fail_before_send(self):
        for changes in ({"reasoning_effort": "high"}, {"model": "gpt-6-luna-other"}, {"model": "gpt-6-astra"}):
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                InformationOpenAIAdapter(config=replace(self.config, **changes))
        adapter, create = self.adapter()
        with self.assertRaises(ConfigurationError):
            adapter.generate(GenerationRequest("information", temperature=.2))
        create.assert_not_called()
        self.assertFalse(adapter.records[0].request_sent)
        self.assertIsNone(adapter.records[0].seed_sent)
        self.assertIsNone(adapter.records[0].temperature_sent)

    def test_refusal_empty_and_truncation_are_not_valid_information(self):
        for response in (chat_response(refusal="refused"), chat_response(""), chat_response(finish="length")):
            adapter, _ = self.adapter(response, max_attempts=1)
            with self.assertRaises(InvalidResponseError):
                adapter.generate(GenerationRequest("information", temperature=0))

    def test_worker_forks_keep_profile_and_fingerprints_private(self):
        parent = InformationOpenAIAdapter(config=self.config)
        workers = [parent.fork(), parent.fork()]
        clients = {id(worker): SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=Mock(return_value=chat_response(fingerprint=f"fp_worker_{index}")))))
            for index, worker in enumerate(workers)}
        with patch.object(InformationOpenAIAdapter, "_get_client", lambda worker: clients[id(worker)]):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda worker: worker.generate(
                    GenerationRequest("information", temperature=0)), workers))
        self.assertEqual(["fp_worker_0", "fp_worker_1"], [item.metadata.system_fingerprint for item in results])
        self.assertEqual([], parent.records)
        self.assertTrue(all(worker.request_scope is parent.request_scope for worker in workers))
        self.assertTrue(all(worker.config == parent.config for worker in workers))

    def test_manifest_contains_effective_profile_and_adapter_code(self):
        from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
        snapshot = SimpleNamespace(source_fingerprint="source", current_input_fingerprint="current")
        adapter = InformationOpenAIAdapter(config=self.config)
        manifest = analysis_manifest(snapshot, InformationAnalysisConfig(), {}, adapter, adapter,
                                     lambda _: {})["manifest"]
        self.assertEqual(2, manifest["version"])
        self.assertEqual({key: manifest["sampling"][key] for key in
            ("api_endpoint", "seed", "requested_temperature", "reasoning_effort")},
            {"api_endpoint": "chat.completions", "seed": 42, "requested_temperature": 0, "reasoning_effort": "none"})
        self.assertIn("adapters/information_llm.py", manifest["source_files"])


@unittest.skipUnless(importlib.util.find_spec("httpx2"), "Pinned OpenAI SDK transport is not installed.")
class InformationProviderSDKTests(unittest.TestCase):
    def test_real_sdk_posts_seed_and_strict_information_schema_to_chat_endpoint(self):
        import httpx2
        from openai import OpenAI
        sent = []
        def handle(request):
            sent.append((request.url.path, json.loads(request.content)))
            return httpx2.Response(200, json=chat_response('{"candidates":[],"issues":[]}'), headers={
                "x-ratelimit-limit-requests": "1000", "x-ratelimit-limit-tokens": "1000000"})
        with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
            with OpenAI(api_key="synthetic", http_client=transport) as client:
                adapter = InformationOpenAIAdapter(config=replace(InformationProviderProfileTests.config,
                    learn_rate_limits=True), client=client)
                for kind in ("information_units", "information_qa", "information_foci"):
                    result = adapter.generate(GenerationRequest("synthetic information",
                        OutputSpec(kind, max_items=4, segment_ids=("v0:s0",)), temperature=0))
        self.assertEqual(3, len(sent))
        for path, body in sent:
            self.assertEqual("/v1/chat/completions", path)
            self.assertEqual((42, 0, "none"), (body["seed"], body["temperature"], body["reasoning_effort"]))
            self.assertFalse(body["store"])
            self.assertEqual(16384, body["max_completion_tokens"])
            self.assertTrue(body["response_format"]["json_schema"]["strict"])
            self.assertIn("candidates", body["response_format"]["json_schema"]["schema"]["properties"])
        self.assertEqual("fp_fixture", result.metadata.system_fingerprint)
        self.assertEqual(900, result.metadata.request_limit)


if __name__ == "__main__":
    unittest.main()
